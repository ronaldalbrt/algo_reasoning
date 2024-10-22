# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Graph algorithm generators.

Currently implements the following:
- Depth-first search (Moore, 1959)
- Breadth-first search (Moore, 1959)
- Topological sorting (Knuth, 1973)
- Articulation points
- Bridges
- Kosaraju's strongly-connected components (Aho et al., 1974)
- Kruskal's minimum spanning tree (Kruskal, 1956)
- Prim's minimum spanning tree (Prim, 1957)
- Bellman-Ford's single-source shortest path (Bellman, 1958)
- Dijkstra's single-source shortest path (Dijkstra, 1959)
- DAG shortest path
- Floyd-Warshall's all-pairs shortest paths (Floyd, 1962)
- Edmonds-Karp bipartite matching (Edmund & Karp, 1972)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name

import torch

from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.probing import graph, array_cat, mask_one

def dfs(A, nb_nodes, *args, **kwargs):
    """Depth-first search (Moore, 1959)."""

    data = CLRSData(algorithm="dfs", *args, **kwargs)

    data.set_inputs({
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    color = torch.zeros(A.size(0), dtype=torch.int32)
    pi = torch.arange(A.size(0))
    d = torch.zeros(A.size(0))
    f = torch.zeros(A.size(0))
    s_prev = torch.arange(A.size(0))
    time = 0

    for s in range(A.size(0)):
        if color[s].item() == 0:
            s_last = s
            u = s
            v = s

            data.increase_hints({
                'pi_h': pi.clone(),
                'color': array_cat(color, 3),
                'd': d.clone(),
                'f': f.clone(),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0)),
                'time': torch.tensor(time)
            })
            while True:
                if color[u].item() == 0 or d[u].item() == 0.0:
                    time += 0.01
                    d[u] = time
                    color[u] = 1

                    data.increase_hints({
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })

                for v in range(A.size(0)):
                    if A[u, v].item() != 0:
                        if color[v].item() == 0:
                            pi[v] = u
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v

                            data.increase_hints({
                                'pi_h': pi.clone(),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time)
                            })
                            break

                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time

                    data.increase_hints({
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd':d.clone(),
                        'f':f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })

                    if s_prev[u].item() == u:
                        assert s_prev[s_last].item() == s_last
                        break
                    pr = s_prev[s_last].item()
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    data.set_outputs({
        'pi': pi.clone()
    })

    return data