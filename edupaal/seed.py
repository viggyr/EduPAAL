"""Exemplar knowledge graph shipped with EduPAAL.

This is *seed content*, not framework: it demonstrates the schema
(Space -> Subject -> Concept -> Topic, recursively decomposable topics,
prerequisite edges that cross concepts/subjects and topic depths) and gives
``examples/quickstart.py`` something concrete. Deployments bring their own
graphs via ``KnowledgeGraph.add_node`` / ``add_prerequisite`` (or a
Mem0-backed loader later).
"""

from __future__ import annotations

from .entities import KnowledgeNode, NodeLevel
from .graph import KnowledgeGraph


def build_seed_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()

    nodes = [
        # Space
        ("science", NodeLevel.SPACE, "Science", None),
        # Subjects
        ("maths", NodeLevel.SUBJECT, "Maths", "science"),
        ("ml", NodeLevel.SUBJECT, "Machine Learning", "science"),
        # Concepts
        ("algebra", NodeLevel.CONCEPT, "Algebra", "maths"),
        ("overfitting", NodeLevel.CONCEPT, "Overfitting", "ml"),
        # Topics
        ("linear-equations", NodeLevel.TOPIC, "Linear Equations", "algebra"),
        ("linearization", NodeLevel.TOPIC, "Linearization", "algebra"),
        ("bias-variance", NodeLevel.TOPIC, "Bias-Variance Tradeoff", "overfitting"),
        ("regularization", NodeLevel.TOPIC, "Regularization", "overfitting"),
        # Sub-topics: Regularization decomposed one level...
        ("l1-l2", NodeLevel.TOPIC, "L1 / L2 Regularization", "regularization"),
        ("dropout", NodeLevel.TOPIC, "Dropout", "regularization"),
        # ...and Dropout decomposed a second level.
        ("dropout-rate", NodeLevel.TOPIC, "Dropout Rate Tuning", "dropout"),
        ("inverted-dropout", NodeLevel.TOPIC, "Inverted Dropout", "dropout"),
    ]
    for nid, level, name, parent in nodes:
        g.add_node(KnowledgeNode(id=nid, level=level, name=name, parent_id=parent))

    # Prerequisites: within-concept, cross-concept/subject, and at sub-topic
    # depth — the graph is a DAG, not a tree.
    g.add_prerequisite("linearization", "linear-equations")
    g.add_prerequisite("bias-variance", "linearization")  # cross-concept/subject
    g.add_prerequisite("regularization", "bias-variance")
    g.add_prerequisite("dropout", "l1-l2")  # sub-topic depth
    g.add_prerequisite("inverted-dropout", "dropout-rate")  # sub-topic depth
    return g
