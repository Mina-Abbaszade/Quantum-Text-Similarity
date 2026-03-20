import torch
import torch.nn.functional as F
from lambeq import PytorchQuantumModel

class CosineSimilarityModel(PytorchQuantumModel):

    def forward(self, diagram_pairs):
        a, b = zip(*diagram_pairs)

        out_a = self.get_diagram_output(a)
        out_b = self.get_diagram_output(b)

        B = out_a.size(0)

        out_a = out_a.view(B, -1)
        out_b = out_b.view(B, -1)

        out_a = F.normalize(out_a, dim=1)
        out_b = F.normalize(out_b, dim=1)

        return F.cosine_similarity(out_a, out_b, dim=1)
