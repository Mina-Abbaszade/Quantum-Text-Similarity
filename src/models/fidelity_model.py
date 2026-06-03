import torch
from lambeq import PytorchQuantumModel

class AmplitudeFidelityModel(PytorchQuantumModel):

    def get_diagram_output(self, diagrams):
        """Return density matrices from tensor contraction."""
        import tensornetwork as tn
        from lambeq.backend.numerical_backend import backend

        diagrams = self._fast_subs(diagrams, self.weights)

        with backend('pytorch'), tn.DefaultBackend('pytorch'):
            results = []

            for d in diagrams:
                nodes, edges = d.to_tn()

                # Ensure uniform dtype
                dominant_dtype = torch.bool
                for node in nodes:
                    dominant_dtype = torch.promote_types(
                        dominant_dtype,
                        node.tensor.dtype
                    )

                for node in nodes:
                    if node.tensor.dtype != dominant_dtype:
                        node.tensor = node.tensor.to(dominant_dtype)

                # Contract tensor network
                result = self._tn_contract(nodes, edges).tensor

                # IMPORTANT:
                # result is already a density matrix
                # do NOT flatten
                # do NOT normalize
                results.append(result)

            return torch.stack(results)

    def matrix_sqrt(self, mat):
        """
        Matrix square root using eigendecomposition.
        Assumes Hermitian positive semi-definite matrices.
        """

        eigvals, eigvecs = torch.linalg.eigh(mat)

        # Numerical stability
        eigvals = torch.clamp(eigvals, min=1e-12)

        sqrt_eigvals = torch.sqrt(eigvals)

        sqrt_diag = torch.diag_embed(sqrt_eigvals).to(eigvecs.dtype)

        return eigvecs @ sqrt_diag @ torch.conj(eigvecs.transpose(-2, -1))

    def density_fidelity(self, rho, sigma):
        """
        Uhlmann fidelity between density matrices.

        rho, sigma: [batch, d, d]
        """

        sqrt_rho = self.matrix_sqrt(rho)

        middle = sqrt_rho @ sigma @ sqrt_rho

        sqrt_middle = self.matrix_sqrt(middle)

        trace = torch.einsum('bii->b', sqrt_middle)

        fidelity = torch.abs(trace) ** 2

        return fidelity.real

    def forward(self, diagram_pairs):

        a, b = zip(*diagram_pairs)

        rho_a = self.get_diagram_output(a)
        rho_b = self.get_diagram_output(b)

        rho_a = rho_a.reshape(len(a), 4, 4)
        rho_b = rho_b.reshape(len(b), 4, 4)

        fidelity = self.density_fidelity(rho_a, rho_b)

        return fidelity
