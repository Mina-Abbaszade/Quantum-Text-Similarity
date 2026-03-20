import torch
from lambeq import PytorchQuantumModel

class AmplitudeFidelityModel(PytorchQuantumModel):

    def get_diagram_output(self, diagrams):
        import tensornetwork as tn
        from lambeq.backend.numerical_backend import backend

        diagrams = self._fast_subs(diagrams, self.weights)

        with backend('pytorch'), tn.DefaultBackend('pytorch'):
            results = []
            for d in diagrams:
                nodes, edges = d.to_tn()

                dominant_dtype = nodes[0].tensor.dtype
                for node in nodes:
                    dominant_dtype = torch.promote_types(
                        dominant_dtype, node.tensor.dtype)

                for node in nodes:
                    if node.tensor.dtype != dominant_dtype:
                        node.tensor = node.tensor.to(dominant_dtype)

                result = self._tn_contract(nodes, edges).tensor
                results.append(result)

            return torch.stack(results)

    def forward(self, diagram_pairs):
        a, b = zip(*diagram_pairs)

        psi_a = self.get_diagram_output(a).view(len(a), -1)
        psi_b = self.get_diagram_output(b).view(len(b), -1)

        psi_a = psi_a / torch.norm(psi_a, dim=1, keepdim=True)
        psi_b = psi_b / torch.norm(psi_b, dim=1, keepdim=True)

        overlap = torch.sum(torch.conj(psi_a) * psi_b, dim=1)
        fidelity = torch.abs(overlap) ** 2

        return fidelity
