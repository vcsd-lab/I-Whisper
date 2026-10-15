import csv
from collections import defaultdict

import torch

from i_whisper.model_int import LayerNorm_int


INT32_MAX = torch.iinfo(torch.int32).max


def _norm_params(hidden_size):
    norm_multiplier = {
        1280: [True, False, False],
        1024: [False, False, False],
        768:  [False, True, True],
        512:  [False, False, False],
        384:  [False, True, True],
    }

    norm_shiftmount = {
        1280: 11,
        1024: 10,
        768: 10,
        512: 9,
        384: 9,
    }

    if hidden_size not in norm_multiplier:
        raise ValueError(
            f"unsupported LayerNorm hidden size: "
            f"{hidden_size}"
        )

    return (
        norm_multiplier[hidden_size],
        norm_shiftmount[hidden_size],
    )


def _mul_by_shift_multiplier(
    x,
    multiplier,
):
    out = x.clone()

    if multiplier[0]:
        out += x >> 1

    if multiplier[1]:
        out += x >> 2

    if multiplier[2]:
        out += x >> 3

    return out


class LayerNormOverflowCollector:
    """
    Collect potential INT32 variance accumulation overflow statistics.

    Statistics are computed from hypothetical unshifted squared
    deviations using INT64 arithmetic. They do not represent
    actual overflow in the production LayerNorm32 kernel.
    """
    def __init__(self):
        self.stats = defaultdict(
            lambda: {
                "calls": 0,
                "overflow_calls": 0,
                "rows": 0,
                "overflow_rows": 0,
                "square_overflow_rows": 0,
                "max_square": 0,
                "max_raw_sum": 0,
            }
        )

        self.handles = []

    @torch.no_grad()
    def _observe(
        self,
        name,
        x,
    ):
        assert x.dtype == torch.int16

        hidden_size = x.shape[-1]

        norm_mul, norm_shift = (
            _norm_params(hidden_size)
        )

        # ----------------------------------------------------
        # Mean:
        # exactly match the current LayerNorm definition.
        # ----------------------------------------------------

        mean = torch.sum(
            x.int(),
            dim=-1,
            keepdim=True,
        ).int()

        mean = (
            _mul_by_shift_multiplier(
                mean,
                norm_mul,
            )
            >> norm_shift
        )

        # ----------------------------------------------------
        # Compute the hypothetical no-shift INT32
        # variance-accumulation workload using INT64
        # only for observation.
        # ----------------------------------------------------

        centered64 = (
            x.to(torch.int64)
            - mean.to(torch.int64)
        )

        square64 = (
            centered64
            * centered64
        )

        raw_sum64 = torch.sum(
            square64,
            dim=-1,
        )

        max_square_per_row = (
            square64.max(
                dim=-1
            ).values
        )

        # Individual square exceeds signed INT32.
        square_overflow = (
            max_square_per_row
            > INT32_MAX
        )

        # Sum of individually valid squared deviations
        # exceeds signed INT32 accumulation range.
        accumulation_overflow = (
            raw_sum64
            > INT32_MAX
        )

        stat = self.stats[name]

        stat["calls"] += 1
        stat["rows"] += raw_sum64.numel()

        overflow_rows = int(
            accumulation_overflow.sum().item()
        )

        stat["overflow_rows"] += (
            overflow_rows
        )

        if overflow_rows > 0:
            stat["overflow_calls"] += 1

        stat["square_overflow_rows"] += int(
            square_overflow.sum().item()
        )

        stat["max_square"] = max(
            stat["max_square"],
            int(
                max_square_per_row.max().item()
            ),
        )

        stat["max_raw_sum"] = max(
            stat["max_raw_sum"],
            int(
                raw_sum64.max().item()
            ),
        )

    def _make_hook(
        self,
        name,
    ):
        def hook(
            module,
            inputs,
        ):
            assert len(inputs) >= 1

            x = inputs[0]

            self._observe(
                name,
                x,
            )

            # Do not return anything:
            # the model input is left untouched.

        return hook

    def attach(
        self,
        model,
    ):
        if self.handles:
            raise RuntimeError(
                "collector is already attached"
            )

        for name, layer in (
            model.named_modules()
        ):
            if isinstance(
                layer,
                LayerNorm_int,
            ):
                handle = (
                    layer.register_forward_pre_hook(
                        self._make_hook(name)
                    )
                )

                self.handles.append(
                    handle
                )

    def detach(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def rows(self):
        output = []

        for name, stat in (
            self.stats.items()
        ):
            rows = stat["rows"]

            overflow_ratio = (
                stat["overflow_rows"]
                / rows
                if rows > 0
                else 0.0
            )

            call_ratio = (
                stat["overflow_calls"]
                / stat["calls"]
                if stat["calls"] > 0
                else 0.0
            )

            output.append(
                {
                    "layer": name,
                    "calls":
                        stat["calls"],
                    "overflow_calls":
                        stat["overflow_calls"],
                    "overflow_call_ratio":
                        call_ratio,
                    "rows":
                        rows,
                    "overflow_rows":
                        stat["overflow_rows"],
                    "overflow_row_ratio":
                        overflow_ratio,
                    "square_overflow_rows":
                        stat[
                            "square_overflow_rows"
                        ],
                    "max_square":
                        stat["max_square"],
                    "max_raw_sum":
                        stat["max_raw_sum"],
                    "max_raw_sum_over_int32":
                        (
                            stat["max_raw_sum"]
                            / INT32_MAX
                        ),
                }
            )

        return sorted(
            output,
            key=lambda x: x["layer"],
        )

    def summary(self):
        total_calls = 0
        overflow_calls = 0
        total_rows = 0
        overflow_rows = 0
        square_overflow_rows = 0
        max_square = 0
        max_raw_sum = 0

        for stat in self.stats.values():
            total_calls += (
                stat["calls"]
            )

            overflow_calls += (
                stat["overflow_calls"]
            )

            total_rows += (
                stat["rows"]
            )

            overflow_rows += (
                stat["overflow_rows"]
            )

            square_overflow_rows += (
                stat[
                    "square_overflow_rows"
                ]
            )

            max_square = max(
                max_square,
                stat["max_square"],
            )

            max_raw_sum = max(
                max_raw_sum,
                stat["max_raw_sum"],
            )

        return {
            "calls":
                total_calls,
            "overflow_calls":
                overflow_calls,
            "overflow_call_ratio":
                (
                    overflow_calls
                    / total_calls
                    if total_calls > 0
                    else 0.0
                ),
            "rows":
                total_rows,
            "overflow_rows":
                overflow_rows,
            "overflow_row_ratio":
                (
                    overflow_rows
                    / total_rows
                    if total_rows > 0
                    else 0.0
                ),
            "square_overflow_rows":
                square_overflow_rows,
            "max_square":
                max_square,
            "max_raw_sum":
                max_raw_sum,
            "max_raw_sum_over_int32":
                (
                    max_raw_sum
                    / INT32_MAX
                    if max_raw_sum > 0
                    else 0.0
                ),
        }

    def write_csv(
        self,
        path,
    ):
        rows = self.rows()

        if not rows:
            raise RuntimeError(
                "no LayerNorm observations collected"
            )

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            path,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)
