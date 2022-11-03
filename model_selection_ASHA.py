from copy import deepcopy
from math import ceil, floor, log
from abc import abstractmethod
from numbers import Integral

import numpy as np
from ._search import BaseSearchCV
from . import ParameterGrid, ParameterSampler
from ..base import is_classifier
from ._split import check_cv, _yields_constant_splits
from ..utils import resample
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples

__all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]


class _SubsampleMetaSplitter:
    """Splitter that subsamples a given fraction of the dataset"""

    def __init__(self, *, base_cv, fraction, subsample_test, random_state):
        self.base_cv = base_cv
        self.fraction = fraction
        self.subsample_test = subsample_test
        self.random_state = random_state

    def split(self, X, y, groups=None):
        for train_idx, test_idx in self.base_cv.split(X, y, groups):
            train_idx = resample(
                train_idx,
                replace=False,
                random_state=self.random_state,
                n_samples=int(self.fraction * train_idx.shape[0]),
            )
            if self.subsample_test:
                test_idx = resample(
                    test_idx,
                    replace=False,
                    random_state=self.random_state,
                    n_samples=int(self.fraction * test_idx.shape[0]),
                )
            yield train_idx, test_idx


def _top_k(results, k, itr):
    # Return the best candidates of a given iteration
    iteration, mean_test_score, params = (
        np.asarray(a)
        for a in (results["iter"], results["mean_test_score"], results["params"])
    )
    iter_indices = np.flatnonzero(iteration == itr)
    scores = mean_test_score[iter_indices]
    # argsort() places NaNs at the end of the array so we move NaNs to the front of the array so the last `k` items are the those with the highest scores.
    sorted_indices = np.roll(np.argsort(scores), np.count_nonzero(np.isnan(scores)))
    return np.array(params[iter_indices][sorted_indices[-k:]])


class BaseSuccessiveHalving(BaseSearchCV):
    """Implements successive halving.

    Ref: Almost optimal exploration in multi-armed bandits, ICML 13, Zohar Karnin, Tomer Koren, Oren Somekh
    """

    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=5,
        verbose=0,
        random_state=None,
        error_score=np.nan,
        return_train_score=True,
        max_resources="auto",
        min_resources="exhaust",
        resource="n_samples",
        factor=3,
        aggressive_elimination=False,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.random_state = random_state
        self.max_resources = max_resources
        self.resource = resource
        self.factor = factor
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination

    def _check_input_parameters(self, X, y, groups):

        if self.scoring is not None and not (
            isinstance(self.scoring, str) or callable(self.scoring)
        ):
            raise ValueError(
                "scoring parameter must be a string, "
                "a callable or None. Multimetric scoring is not "
                "supported."
            )

        # We need to enforce that successive calls to cv.split() yield the same splits
        if not _yields_constant_splits(self._checked_cv_orig):
            raise ValueError(
                "The cv parameter must yield consistent folds across "
                "calls to split(). Set its random_state to an int, or set "
                "shuffle=False."
            )

        if (
            self.resource != "n_samples"
            and self.resource not in self.estimator.get_params()
        ):
            raise ValueError(
                f"Cannot use resource={self.resource} which is not supported "
                f"by estimator {self.estimator.__class__.__name__}"
            )

        if isinstance(self.max_resources, str) and self.max_resources != "auto":
            raise ValueError(
                "max_resources must be either 'auto' or a positive integer"
            )
        if self.max_resources != "auto" and (
            not isinstance(self.max_resources, Integral) or self.max_resources <= 0
        ):
            raise ValueError(
                "max_resources must be either 'auto' or a positive integer"
            )

        if self.min_resources not in ("smallest", "exhaust") and (
            not isinstance(self.min_resources, Integral) or self.min_resources <= 0
        ):
            raise ValueError(
                "min_resources must be either 'smallest', 'exhaust', "
                "or a positive integer "
                "no greater than max_resources."
            )

        if isinstance(self, HalvingRandomSearchCV):
            if self.min_resources == self.n_candidates == "exhaust":
                # for n_candidates=exhaust to work, we need to know what min_resources is. Similarly min_resources=exhaust needs to know the actual number of candidates.
                raise ValueError(
                    "n_candidates and min_resources cannot be both set to 'exhaust'."
                )
            if self.n_candidates != "exhaust" and (
                not isinstance(self.n_candidates, Integral) or self.n_candidates <= 0
            ):
                raise ValueError(
                    "n_candidates must be either 'exhaust' or a positive integer"
                )

        self.min_resources_ = self.min_resources
        if self.min_resources_ in ("smallest", "exhaust"):
            if self.resource == "n_samples":
                n_splits = self._checked_cv_orig.get_n_splits(X, y, groups)
                # please see https://gph.is/1KjihQe for a justification
                magic_factor = 2
                self.min_resources_ = n_splits * magic_factor
                if is_classifier(self.estimator):
                    y = self._validate_data(X="no_validation", y=y)
                    check_classification_targets(y)
                    n_classes = np.unique(y).shape[0]
                    self.min_resources_ *= n_classes
            else:
                self.min_resources_ = 1
            # if 'exhaust', min_resources_ might be set to a higher value later in _run_search

        self.max_resources_ = self.max_resources
        if self.max_resources_ == "auto":
            if not self.resource == "n_samples":
                raise ValueError(
                    "resource can only be 'n_samples' when max_resources='auto'"
                )
            self.max_resources_ = _num_samples(X)

        if self.min_resources_ > self.max_resources_:
            raise ValueError(
                f"min_resources_={self.min_resources_} is greater "
                f"than max_resources_={self.max_resources_}."
            )

        if self.min_resources_ == 0:
            raise ValueError(
                f"min_resources_={self.min_resources_}: you might have passed "
                "an empty dataset X."
            )

        if not isinstance(self.refit, bool):
            raise ValueError(
                f"refit is expected to be a boolean. Got {type(self.refit)} instead."
            )

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Custom refit callable to return the index of the best candidate.

        We want the best candidate out of the last iteration. By default
        BaseSearchCV would return the best candidate out of all iterations.
        """
        last_iter = np.max(results["iter"])
        last_iter_indices = np.flatnonzero(results["iter"] == last_iter)

        test_scores = results["mean_test_score"][last_iter_indices]
        # If all scores are NaNs there is no way to pick between them, so we (arbitrarily) declare the zero'th entry the best one
        if np.isnan(test_scores).all():
            best_idx = 0
        else:
            best_idx = np.nanargmax(test_scores)

        return last_iter_indices[best_idx]

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters."""
        self._checked_cv_orig = check_cv(
            self.cv, y, classifier=is_classifier(self.estimator)
        )

        self._check_input_parameters(
            X=X,
            y=y,
            groups=groups,
        )

        self._n_samples_orig = _num_samples(X)

        super().fit(X, y=y, groups=groups, **fit_params)

        # Set best_score_: BaseSearchCV does not set it, as refit is a callable
        self.best_score_ = self.cv_results_["mean_test_score"][self.best_index_]

        return self

    def _run_search(self, evaluate_candidates):
        candidate_params = self._generate_candidate_params()

        if self.resource != "n_samples" and any(
            self.resource in candidate for candidate in candidate_params
        ):
            # Can only check this now since we need the candidates list
            raise ValueError(
                f"Cannot use parameter {self.resource} as the resource since "
                "it is part of the searched parameters."
            )

        # n_required_iterations is the number of iterations needed so that the last iterations evaluates less than `factor` candidates.
        n_required_iterations = 1 + floor(log(len(candidate_params), self.factor))

        if self.min_resources == "exhaust":
            # To exhaust the resources, we want to start with the biggest min_resources possible so that the last (required) iteration uses as many resources as possible
            last_iteration = n_required_iterations - 1
            self.min_resources_ = max(
                self.min_resources_,
                self.max_resources_ // self.factor**last_iteration,
            )

        # n_possible_iterations is the number of iterations that we can actually do starting from min_resources and without exceeding max_resources. Depending on max_resources and the number of candidates, this may be higher or smaller than n_required_iterations.
        n_possible_iterations = 1 + floor(
            log(self.max_resources_ // self.min_resources_, self.factor)
        )

        if self.aggressive_elimination:
            n_iterations = n_required_iterations
        else:
            n_iterations = min(n_possible_iterations, n_required_iterations)

        if self.verbose:
            print(f"n_iterations: {n_iterations}")
            print(f"n_required_iterations: {n_required_iterations}")
            print(f"n_possible_iterations: {n_possible_iterations}")
            print(f"min_resources_: {self.min_resources_}")
            print(f"max_resources_: {self.max_resources_}")
            print(f"aggressive_elimination: {self.aggressive_elimination}")
            print(f"factor: {self.factor}")

        self.n_resources_ = []
        self.n_candidates_ = []

        for itr in range(n_iterations):

            power = itr  # default
            if self.aggressive_elimination:
                # this will set n_resources to the initial value (i.e. the value of n_resources at the first iteration) for as many iterations as needed (while candidates are being eliminated), and then go on as usual.
                power = max(0, itr - n_required_iterations + n_possible_iterations)

            n_resources = int(self.factor**power * self.min_resources_)
            # guard, probably not needed
            n_resources = min(n_resources, self.max_resources_)
            self.n_resources_.append(n_resources)

            n_candidates = len(candidate_params)
            self.n_candidates_.append(n_candidates)

            if self.verbose:
                print("-" * 10)
                print(f"iter: {itr}")
                print(f"n_candidates: {n_candidates}")
                print(f"n_resources: {n_resources}")

            if self.resource == "n_samples":
                # subsampling will be done in cv.split()
                cv = _SubsampleMetaSplitter(
                    base_cv=self._checked_cv_orig,
                    fraction=n_resources / self._n_samples_orig,
                    subsample_test=True,
                    random_state=self.random_state,
                )

            else:
                # Need copy so that the n_resources of next iteration does not overwrite
                candidate_params = [c.copy() for c in candidate_params]
                for candidate in candidate_params:
                    candidate[self.resource] = n_resources
                cv = self._checked_cv_orig

            more_results = {
                "iter": [itr] * n_candidates,
                "n_resources": [n_resources] * n_candidates,
            }

            results = evaluate_candidates(
                candidate_params, cv, more_results=more_results
            )

            n_candidates_to_keep = ceil(n_candidates / self.factor)
            candidate_params = _top_k(results, n_candidates_to_keep, itr)

        self.n_remaining_candidates_ = len(candidate_params)
        self.n_required_iterations_ = n_required_iterations
        self.n_possible_iterations_ = n_possible_iterations
        self.n_iterations_ = n_iterations

    @abstractmethod
    def _generate_candidate_params(self):
        pass

    def _more_tags(self):
        tags = deepcopy(super()._more_tags())
        tags["_xfail_checks"].update(
            {
                "check_fit2d_1sample": (
                    "Fail during parameter check since min/max resources requires"
                    " more samples"
                ),
            }
        )
        return tags


class HalvingGridSearchCV(BaseSuccessiveHalving):
    """Search over specified parameter values with successive halving.

    The search strategy starts evaluating all the candidates with a small
    amount of resources and iteratively selects the best candidates, using
    more and more resources.
    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        self.param_grid = param_grid

    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)


class HalvingRandomSearchCV(BaseSuccessiveHalving):
    """Randomized search on hyper parameters.

    The search strategy starts evaluating all the candidates with a small
    amount of resources and iteratively selects the best candidates, using more
    and more resources.

    The candidates are sampled at random from the parameter space and the
    number of sampled candidates is determined by ``n_candidates``.
    """

    _required_parameters = ["estimator", "param_distributions"]

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_candidates="exhaust",
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="smallest",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates

    def _generate_candidate_params(self):
        n_candidates_first_iter = self.n_candidates
        if n_candidates_first_iter == "exhaust":
            # This will generate enough candidate so that the last iteration uses as much resources as possible
            n_candidates_first_iter = self.max_resources_ // self.min_resources_
        return ParameterSampler(
            self.param_distributions,
            n_candidates_first_iter,
            random_state=self.random_state,
        )
