"""The expression a tracked term carries, its rendering, and the depth a rendering shows."""

from __future__ import annotations

import copy
import pickle
import warnings

import jax
import jax.numpy as jnp
import pytest

import probpipe
from probpipe import (
    Function,
    Normal,
    NumericArray,
    NumericArraySpec,
    Record,
    RecordBatch,
    conditional_distribution,
    function,
    sample,
    workflow_run,
)
from probpipe.core._expression import (
    _STORED_DEPTH,
    Applied,
    Conditioned,
    Draw,
    Indexed,
    Named,
    Operator,
    Product,
    Selected,
    Signature,
    Summary,
    constant,
)
from probpipe.core._fingerprint import fingerprint


def _prior() -> Normal:
    return Normal("mu", 0.0, 1.0, label="prior")


def _named(label: str, *components: str) -> Named:
    return Named(label, Signature(components))


class TestNodes:
    """The nodes hold strings and nodes, and they compare by value."""

    def test_a_node_compares_and_hashes_by_value(self):
        assert _named("prior", "mu") == _named("prior", "mu")
        assert hash(_named("prior", "mu")) == hash(_named("prior", "mu"))
        assert _named("prior", "mu") != _named("prior", "tau")

    def test_a_node_is_frozen(self):
        with pytest.raises(AttributeError):
            _named("prior", "mu").label = "other"  # type: ignore[misc]

    def test_a_product_of_products_is_flat(self):
        product = Product((Product((Named("a"), Named("b"))), Named("c")))
        assert product.factors == (Named("a"), Named("b"), Named("c"))

    def test_conditioning_a_conditioned_law_merges_into_one_node(self):
        base = _named("model", "y", "mu")
        twice = Conditioned(Conditioned(base, ("y",)), ("z", "y"))
        assert twice == Conditioned(base, ("y", "z"))

    def test_an_unknown_summary_raises(self):
        with pytest.raises(ValueError, match="unknown summary kind"):
            Summary("median", Named("x"))


class TestRendering:
    """The label and the notation of each node, grouped by the rules of design II.4."""

    @pytest.mark.parametrize(
        ("expression", "label", "notation"),
        [
            pytest.param(_named("prior", "mu"), "prior", "prior(mu)", id="named"),
            pytest.param(
                Product((Named("lik", Signature(("y",), ("mu",))), _named("prior", "mu"))),
                "lik·prior",
                "lik(y | mu)·prior(mu)",
                id="product",
            ),
            pytest.param(
                Conditioned(_named("model", "y", "mu"), ("y",), Signature(("mu",))),
                "model",
                "model(mu; y)",
                id="conditioned",
            ),
            pytest.param(
                Selected(_named("model", "y", "mu"), ("y",), Signature(("y",))),
                "model",
                "model(y)",
                id="selected",
            ),
            pytest.param(
                Conditioned(Product((Named("lik"), Named("prior"))), ("y",), Signature(("mu",))),
                "lik·prior",
                "(lik·prior)(mu; y)",
                id="conditioned-product",
            ),
            pytest.param(
                Applied("f", (Draw(("beta",), Conditioned(Named("m"), ("y",))),)),
                "f",
                "f(beta ~ m; y)",
                id="applied",
            ),
            pytest.param(
                Applied("log_prob", (_named("g", "g"), Draw(("q",), _named("q", "q")))),
                "log_prob",
                "log_prob(g(g), q ~ q)",
                id="applied-to-a-law",
            ),
            pytest.param(
                Indexed(
                    Applied("f", (Draw(("mu",), Named("m")), Named("tau"))),
                    "tau=3",
                    element=Applied("f", (Draw(("mu",), Named("m")), constant(4.0))),
                ),
                "f[tau=3]",
                "f(mu ~ m, 4.0)",
                id="element-of-a-lifted-batch",
            ),
            pytest.param(
                Indexed(Applied("f", (Draw(("mu",), Named("m")), Named("tau"))), "tau=1:3"),
                "f[tau=1:3]",
                "f(mu ~ m, tau)[tau=1:3]",
                id="selection-of-a-lifted-batch",
            ),
        ],
    )
    def test_a_law_renders_its_label_and_its_notation(self, expression, label, notation):
        assert (expression.render_label(), expression.render_notation()) == (label, notation)

    @pytest.mark.parametrize(
        ("expression", "label"),
        [
            pytest.param(Draw(("mu",), _named("prior", "mu")), "mu ~ prior", id="one-component"),
            pytest.param(
                Draw(("y", "mu"), _named("model", "y", "mu")),
                "(y, mu) ~ model",
                id="several-components",
            ),
            pytest.param(
                Draw(("mu",), Conditioned(_named("model", "y", "mu"), ("y",))),
                "mu ~ model; y",
                id="fixed-paths",
            ),
            pytest.param(
                Draw(("y", "mu"), Product((Named("lik"), Named("prior")))),
                "(y, mu) ~ lik·prior",
                id="tilde-binds-most-loosely",
            ),
            pytest.param(Summary("log", _named("prior", "mu")), "log prior(mu)", id="score"),
            pytest.param(
                Summary("log", Product((_named("a", "a"), _named("b", "b")))),
                "log (a(a)·b(b))",
                id="score-of-a-product",
            ),
            pytest.param(Summary("density", _named("prior", "mu")), "prior(mu)", id="density"),
            pytest.param(
                Summary("E", Draw(("y", "mu"), _named("model", "y", "mu"))),
                "E[(y, mu) ~ model]",
                id="mean",
            ),
            pytest.param(
                Summary("Q", Draw(("mu",), _named("prior", "mu"))), "Q[mu ~ prior]", id="quantile"
            ),
            pytest.param(
                Summary("E", Draw(("p",), Applied("f", (Draw(("b",), Named("m")),)))),
                "E[f(b ~ m)]",
                id="mean-of-a-lifted-law",
            ),
            pytest.param(
                Summary(
                    "E",
                    Draw(
                        ("f",),
                        Indexed(
                            Applied("f", (Named("tau"),)),
                            "tau=1",
                            element=Applied("f", (constant(2.0),)),
                        ),
                    ),
                ),
                "E[f(2.0)]",
                id="mean-of-an-element-of-a-lifted-batch",
            ),
            pytest.param(Operator("*", (constant(2), Named("effect"))), "2 * effect", id="binary"),
            pytest.param(
                Operator("+", (Operator("*", (Named("a"), Named("b"))), constant(1.0))),
                "(a * b) + 1.0",
                id="nested-operator",
            ),
            pytest.param(Operator("-", (Named("x"),)), "-x", id="prefix"),
            pytest.param(Operator("abs", (Named("x"),)), "abs(x)", id="call"),
            pytest.param(
                Indexed(Draw(("mu",), Named("prior")), "sample=0"),
                "(mu ~ prior)[sample=0]",
                id="draw-in-a-selection",
            ),
            pytest.param(
                Indexed(Summary("log", _named("prior", "mu")), "sample=0"),
                "(log prior(mu))[sample=0]",
                id="score-in-a-selection",
            ),
            pytest.param(
                Indexed(Product((Named("x"), Named("y"))), "sample=0:2"),
                "(x·y)[sample=0:2]",
                id="product-in-a-selection",
            ),
        ],
    )
    def test_a_value_renders_in_full_as_its_label(self, expression, label):
        assert expression.render_label() == expression.render_notation() == label

    def test_a_terms_own_signature_replaces_the_recorded_one(self):
        expression = Conditioned(_named("model", "y", "mu"), ("y",))
        assert expression.render_notation() == "model"
        assert expression.render_notation(Signature(("mu",))) == "model(mu; y)"

    def test_labels_join_associatively_and_group_other_labels(self):
        product = Product((Named("lik·prior"), Named("model | y"), Named("my prior")))
        assert product.render_label() == "lik·prior·(model | y)·[my prior]"


class TestFixedPaths:
    """A law's fixed paths are read from its expression."""

    def test_each_node_holds_its_bases_fixed_paths(self):
        named = _named("post", "mu")
        assert named.fixed_paths() == ()
        conditioned = Conditioned(Conditioned(named, ("y",)), ("z", "y"))
        assert conditioned.fixed_paths() == ("y", "z")
        assert Selected(conditioned, ("mu",)).fixed_paths() == ("y", "z")
        assert Indexed(conditioned, "row=0").fixed_paths() == ("y", "z")
        assert Product((conditioned,)).fixed_paths() == ()

    def test_with_fixed_adds_only_the_paths_not_held(self):
        base = Conditioned(Named("model"), ("y",))
        assert base.with_fixed(("y",)) is base
        assert base.with_fixed(("z/a", "y")).fixed_paths() == ("y", "z/a")

    def test_core_strips_conditionings_and_selections(self):
        product = Product((Named("a"), Named("b")))
        assert Selected(Conditioned(product, ("y",)), ("a",)).core() is product


class TestDepth:
    """A rendering shows at most ``notation_config.max_depth`` nested levels."""

    @staticmethod
    def _chain(steps: int):
        expression = Named("x")
        for _ in range(steps):
            expression = Operator("+", (expression, constant(1)))
        return expression

    def test_a_shallow_rendering_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._chain(2).render_label(warn=True) == "(x + 1) + 1"

    def test_a_deeper_rendering_collapses_a_value_to_an_ellipsis_and_warns(self):
        probpipe.notation_config.max_depth = 2
        with pytest.warns(UserWarning, match="notation_config.max_depth=2"):
            assert self._chain(3).render_label(warn=True) == "(… + 1) + 1"

    def test_a_collapsed_law_or_function_shows_its_label(self):
        probpipe.notation_config.max_depth = 1
        lifted = Summary("E", Draw(("p",), Applied("f", (Draw(("b",), Named("m")),))))
        with pytest.warns(UserWarning, match="max_depth"):
            assert lifted.render_label(warn=True) == "E[f]"
        product = Draw(("a", "b"), Product((_named("a", "a"), _named("b", "b"))))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert product.render_label(warn=True) == "(a, b) ~ a·b"

    def test_a_law_reads_in_full_at_any_depth(self):
        """A conditioning or a selection nests no level, so a draw from one reads in full."""
        probpipe.notation_config.max_depth = 2
        draw = Draw(("mu",), Selected(Conditioned(Named("model"), ("y",)), ("mu",)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert Summary("E", draw).render_label(warn=True) == "E[mu ~ model; y]"

    def test_raising_the_depth_shows_the_collapsed_levels(self):
        deep = self._chain(10)
        with pytest.warns(UserWarning):
            assert "…" in deep.render_label(warn=True)
        probpipe.notation_config.max_depth = 12
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert "…" not in deep.render_label(warn=True)

    def test_a_stored_tree_is_bounded(self):
        """A long derivation stores a tree of bounded depth, which copies and pickles."""
        deep = self._chain(3 * _STORED_DEPTH)
        assert deep.depth <= _STORED_DEPTH + 1
        assert pickle.loads(pickle.dumps(deep)) == deep

    def test_a_term_derived_in_a_long_loop_pickles(self):
        value = NumericArray("x", jnp.asarray(1.0))
        for _ in range(300):
            value = value + 1.0
        restored = pickle.loads(pickle.dumps(value))
        assert restored.label == value.label
        assert float(restored) == 301.0

    def test_a_rendering_that_stores_a_label_does_not_warn(self):
        probpipe.notation_config.max_depth = 2
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._chain(3).render_label() == "(… + 1) + 1"


class TestTheWarningFiresWhenATermIsShown:
    """A collapsed rendering warns when a term is shown, and never while terms are computed."""

    @staticmethod
    def _looped() -> NumericArray:
        with workflow_run(seed=0):
            value = sample(_prior())
        for _ in range(20):
            value = value + 1.0
        return value

    def test_deriving_a_term_in_a_loop_and_reading_its_label_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = self._looped()
            assert "…" in value.label

    def test_the_repr_of_a_collapsed_label_warns_once(self):
        value = self._looped()
        with pytest.warns(UserWarning, match="notation_config.max_depth=8") as caught:
            repr(value)
        assert len([w for w in caught if "max_depth" in str(w.message)]) == 1

    def test_the_warning_names_the_line_that_shows_the_term(self):
        value = self._looped()
        with pytest.warns(UserWarning, match="max_depth") as caught:
            repr(value)
        assert caught[0].filename == __file__

    def test_the_str_and_notation_of_a_collapsed_law_warn(self):
        @function
        def f(mu: jax.Array) -> jax.Array:
            return mu + 1.0

        with workflow_run(seed=0):
            law = f.with_options(n_broadcast_samples=4)(_prior())
        probpipe.notation_config.max_depth = 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert law.label == "f"
        with pytest.warns(UserWarning, match="max_depth=1"):
            assert str(law) == "f(…)"
        with pytest.warns(UserWarning, match="max_depth=1"):
            assert law.notation == "f(…)"


class TestTheLabelIsTheExpressionsLabel:
    """A term's label always equals its expression's label: one source of truth."""

    @staticmethod
    def _terms():
        prior = _prior()
        kernel = conditional_distribution(
            lambda beta: Normal("y", beta, 1.0),
            given_spec={"beta": NumericArraySpec(())},
            label="glm",
        )
        record = Record("r", {"a": 1.0, "b": 2.0})
        batch = RecordBatch.stack([record, record], level_name="row", label="rows")
        joint = kernel * prior.with_path_names(mu="beta")
        return [
            prior,
            prior.with_label("p"),
            kernel,
            joint,
            joint.with_label("model"),
            joint["y"],
            joint["beta"],
            record,
            record["a"],
            batch,
            batch[0],
            batch[0:1],
            NumericArray("x", jnp.zeros(2)) * 2.0,
            Function("predict", lambda x: x),
        ]

    def test_every_term_carries_the_label_of_its_expression(self):
        for term in self._terms():
            assert term.label == term._expression.render_label(), term

    def test_with_label_replaces_the_expression_with_the_label(self):
        """A user's label hides the derivation, and the paths the law holds fixed stay."""
        model = _prior() * Normal("y", 0.0, 1.0)
        derived = model._with_expression(Conditioned(model._expression, ("y",)))
        relabeled = derived.with_label("posterior")
        assert relabeled._expression == Conditioned(
            Named("posterior", Signature(("mu", "y"))), ("y",)
        )
        assert relabeled.notation == "posterior(mu, y; y)"
        assert relabeled.provenance.operation == "with_label"

    def test_embedding_a_law_records_its_signature(self):
        kernel = conditional_distribution(
            lambda beta: Normal("y", beta, 1.0),
            given_spec={"beta": NumericArraySpec(())},
            label="glm",
        )
        assert kernel._embedded_expression() == Named("glm", Signature(("y",), ("beta",)))
        assert kernel._expression == Named("glm")


class TestIndependenceFromComputation:
    """Fingerprints omit the expression, and copies and pickles keep it."""

    def test_the_expression_leaves_the_fingerprint_unchanged(self):
        prior = _prior()
        conditioned = prior._with_expression(Conditioned(prior._expression, ("y",)))
        assert fingerprint(conditioned) == fingerprint(prior)

    @pytest.mark.parametrize("round_trip", [copy.copy, copy.deepcopy, pickle.dumps])
    def test_copies_and_pickles_keep_the_expression(self, round_trip):
        prior = _prior()
        conditioned = prior._with_expression(Conditioned(prior._expression, ("y",)))
        restored = round_trip(conditioned)
        if isinstance(restored, bytes):
            restored = pickle.loads(restored)
        assert restored._expression == conditioned._expression
        assert restored.notation == "prior(mu; y)"
