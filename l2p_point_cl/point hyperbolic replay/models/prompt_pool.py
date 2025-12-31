import torch
import torch.nn as nn
from methods.poincareball import PoincareBall

class HyperbolicPromptPool(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 c=1.0, num_tasks=20, prompts_per_task=3, map_scale=0.1, selection_metric='hyperbolic'):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.ball = PoincareBall(dim=embed_dim, c=c) # Hyperbolic Space
        self.num_tasks = num_tasks
        self.prompts_per_task = prompts_per_task
        self.map_scale = map_scale
        self.selection_metric = selection_metric

        if self.prompt_pool:
            if pool_size is None:
                # Progressive Strategy: Total Pool = Tasks * Prompts/Task
                self.pool_size = num_tasks * prompts_per_task
            else:
                self.pool_size = pool_size
            
            prompt_pool_shape = (self.pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (self.pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def map_to_ball(self, x, dim=1, scale=None):
        """Map Euclidean vectors to the Poincare ball via expmap0 after normalization."""
        if scale is None:
            scale = self.map_scale
        x_norm = self.l2_normalize(x, dim=dim)
        x_scaled = x_norm * scale
        x_ball = self.ball.expmap0(x_scaled)
        return self.ball.proju0(x_ball)

    def compute_similarity(self, query, keys):
        if self.selection_metric == 'hyperbolic':
            query_ball = self.map_to_ball(query, dim=1)
            keys_ball = self.map_to_ball(keys, dim=1)
            query_expanded = query_ball.unsqueeze(1)
            keys_expanded = keys_ball.unsqueeze(0)
            distance = self.ball.dist(query_expanded, keys_expanded, keepdim=False) # [B, Pool]
            similarity = -1 * distance
            return similarity, distance, query_ball, keys_ball

        if self.selection_metric == 'cosine':
            query_norm = self.l2_normalize(query, dim=1)
            keys_norm = self.l2_normalize(keys, dim=1)
            similarity = torch.matmul(query_norm, keys_norm.T)
            distance = -similarity
            return similarity, distance, query_norm, keys_norm

        raise ValueError(f"Unsupported selection_metric: {self.selection_metric}")
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, task_id=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            similarity, distance, x_embed_norm, prompt_key_norm = self.compute_similarity(
                x_embed_mean, self.prompt_key
            )

            PoolSize = prompt_key_norm.shape[0]

            # --- PROGRESSIVE STRATEGY MASKING ---
            if task_id is not None:
                # Mask future prompts
                valid_limit = (task_id + 1) * self.prompts_per_task
                if valid_limit < PoolSize:
                    # Set distance to infinity for future prompts so they are never selected
                    distance[:, valid_limit:] = float('inf')
            
            # L2P selects Top-K *Closest* -> Smallest Distance
            # similarity = -distance (so Max(-dist) = Min(dist))
            similarity = -1 * distance
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                
                # FIX: Sort indices to ensure permutation invariance for the classifier
                # Otherwise [Prompt A, Prompt B] and [Prompt B, Prompt A] look different to Linear layer
                idx, _ = torch.sort(idx, dim=1)
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_key_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            
            # Hyperbolic Pull Constraint
            # Pull selected keys closer to the query
            # We can return the distance calculated earlier
            
            # Gather distances for selected prompts
            # idx: [B, top_k]
            # distance: [B, Pool_Size]
            selected_distances = torch.gather(distance, 1, idx) # [B, top_k]
            
            # Reduce Sim -> Reduce Dist (We want to minimize this)
            reduce_dist = torch.sum(selected_distances) / x_embed.shape[0] # Scalar
            out['reduce_sim'] = reduce_dist # Note: The trainer should know this is distance and MINIMIZE it.

            if task_id is not None:
                start_idx = task_id * self.prompts_per_task
                end_idx = (task_id + 1) * self.prompts_per_task
                prompt_slice = self.prompt[start_idx:end_idx]  # T, L, C
                if prompt_slice.numel() > 0:
                    prompt_flat = prompt_slice.reshape(prompt_slice.shape[0], -1)
                    prompt_flat = self.l2_normalize(prompt_flat, dim=1)
                    gram = torch.matmul(prompt_flat, prompt_flat.T)
                    identity = torch.eye(gram.size(0), device=gram.device)
                    out['orth_loss'] = torch.norm(gram - identity, p='fro')
                else:
                    out['orth_loss'] = torch.tensor(0.0, device=x_embed.device)

        else:
            # No pool, just static or learnable prompt without selection
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
