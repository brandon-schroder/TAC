from elements import *

# =============================================================================
# 7. TURBO SYSTEM
# =============================================================================
class TurboSystem:
    """
    Assembles and executes the element network.

    Execution pattern (local implicit, matching OTAC):
      Seed → for each element: execute (subelements implicit inside) →
      propagate → check convergence → repeat.

    Subelements no longer registered here — they live inside each BladeRow
    and are called automatically inside residuals().
    """

    def __init__(self, name: str):
        self.name         = name
        self.elements:    list[Element] = []
        self.connections: list[tuple]   = []

    def add_element(self, element: Element):
        self.elements.append(element)

    def connect(self, port_out: FlowStation, port_in: FlowStation):
        self.connections.append((port_out, port_in))

    def _apply_connections(self):
        for port_out, port_in in self.connections:
            link_ports(port_out, port_in)

    def run(self, omega_map: dict, n_iter: int = 10, tol: float = 1.0):
        print(f"\n{'='*52}")
        print(f"  TurboSystem : {self.name}  |  Running...")
        print(f"{'='*52}")

        # Seed: push boundary conditions through connections before iteration 1
        for elem in self.elements:
            if isinstance(elem, InletBoundary):
                elem.execute()
        self._apply_connections()

        prev_pts = {}

        for iteration in range(n_iter):
            print(f"\n--- Iteration {iteration + 1} ---")

            for elem in self.elements:
                omega = omega_map.get(elem.name, 0.0)
                elem.execute(omega=omega)
                # Propagate immediately so next element's Fl_I is always current
                self._apply_connections()

            # Convergence check on exit Pt of every BladeRow
            converged = True
            for elem in self.elements:
                if isinstance(elem, BladeRow):
                    pt_now = elem.Fl_O.pt
                    if elem.name in prev_pts:
                        if abs(pt_now - prev_pts[elem.name]) > tol:
                            converged = False
                    else:
                        converged = False
                    prev_pts[elem.name] = pt_now

            if converged and iteration > 0:
                print(f"\n  ✓ Converged after {iteration + 1} iterations.")
                break
        else:
            print(f"\n  ⚠ Reached max iterations ({n_iter}) without full convergence.")

    def report(self):
        print(f"\n{'='*52}")
        print(f"  SYSTEM RESULTS : {self.name}")
        print(f"{'='*52}")
        for elem in self.elements:
            elem.Fl_O.report()


