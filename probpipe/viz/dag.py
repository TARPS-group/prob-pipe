from graphviz import Digraph
from ..core.node import WorkflowNode


def visualize_module_dag(module):
    """
    Visualize a ProbPipe ModuleNode as a clustered DAG.

    - Child nodes: ellipses (outside cluster)
    - Module: cluster
    - Workflow nodes: boxes (inside cluster)
    - Edges: child_node -> workflow_node (dependency semantics)
    """

    dot = Digraph(
        name=module.__class__.__name__,
        graph_attr={
            "rankdir": "LR",
            "fontsize": "12",
            "fontname": "Helvetica",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
        },
    )

    # -------------------------
    # Child nodes (outside)
    # -------------------------
    for name in module._child_nodes:
        dot.node(
            name,
            label=name,
            shape="ellipse",
            style="filled",
            fillcolor="#E8E8E8",
        )

    # -------------------------
    # Module cluster
    # -------------------------
    with dot.subgraph(name=f"cluster_{module.__class__.__name__}") as cluster:
        cluster.attr(
            label=module.__class__.__name__,
            style="rounded",
            color="#4F81BD",
            fontname="Helvetica-Bold",
            fontsize="12",
        )

        # Workflow nodes inside the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not isinstance(attr, WorkflowNode):
                continue

            wf_name = attr._name  # e.g. PM25ForecastingModule.fit
            wf_label = wf_name.split(".")[-1]

            cluster.node(
                wf_name,
                label=wf_label,
                shape="box",
                style="filled",
                fillcolor="#C6DBEF",
            )

    # -------------------------
    # Dependency edges
    # -------------------------
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not isinstance(attr, WorkflowNode):
            continue

        wf_name = attr._name

        for child_name in attr._child_nodes:
            if child_name not in module._child_nodes:
                raise ValueError(
                    f"Workflow '{wf_name}' references unknown child node '{child_name}'"
                )

            dot.edge(child_name, wf_name)

    return dot