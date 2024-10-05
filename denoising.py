"""
Utilities for denoising/corrupting a sequence of tokens for the UL2 objectives.

Most of this is taken directly from t5.data.preprocessors with the only changes
being to support pytorch instead of tensorflow:

https://github.com/google-research/text-to-text-transfer-transformer/blob/69cf3b5913fe006b070cc37eef751de51b59dc6f/t5/data/preprocessors.py
"""

import torch


def stateless_shuffle(value):
    """see seqio.stateless_shuffle for reference"""
    flat_value = value.view(-1)
    indices = torch.randperm(flat_value.size(0))
    flat_shuffle = flat_value[indices]

    return flat_shuffle.view(value.size())


def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """see tf.math.segment_sum for reference"""
    max_segment_id = segment_ids.max().item() + 1
    result = torch.zeros(
        max_segment_id, *data.shape[1:], dtype=data.dtype, device=data.device
    )
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """see tf.math.unsorted_segment_sum for reference"""
    assert (
        data.shape[0] == segment_ids.shape[0]
    ), "data and segment_ids must have the same size in the first dimension"
    assert segment_ids.dtype == torch.long, "segment_ids must be of type torch.long"
    assert num_segments > 0, "num_segments must be positive"

    result = torch.zeros(
        num_segments, *data.shape[1:], dtype=data.dtype, device=data.device
    )
    result.scatter_add_(0, segment_ids, data)
    return result


def random_spans_noise_mask(length, noise_density, mean_noise_span_length=3.0):
    """implementation of t5.data.preprocessors.random_spans_noise_mask but in pytorch"""
    if noise_density == 0.0:
        return torch.zeros(length, dtype=torch.bool)

    orig_length = length

    length = max(length, 2)

    num_noise_tokens = int(round(float(length) * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the nonnoise spans
    def _random_segmentation(num_items, num_segments):
        first_in_segment = torch.nn.functional.pad(
            stateless_shuffle(
                (torch.arange(num_items - 1) < num_segments - 1).to(torch.long)
            ),
            (1, 0),
        )

        segment_id = torch.cumsum(first_in_segment, dim=0)
        segment_length = segment_sum(torch.ones_like(segment_id), segment_id)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = torch.stack(
        [nonnoise_span_lengths, noise_span_lengths], dim=1
    ).reshape(num_noise_spans * 2)
    span_starts = torch.cumsum(interleaved_span_lengths, dim=0)[:-1]
    span_start_indicator = unsorted_segment_sum(
        torch.ones_like(span_starts), span_starts, length
    )
    span_num = torch.cumsum(span_start_indicator, dim=0)
    is_noise = span_num % 2 == 1

    mask = is_noise[:orig_length]

    return mask


def random_prefix_noise_mask(length, noise_density):
    """implementation of t5.data.preproccessors.random_prefix_noise_mask but in pytorch"""
    if noise_density > 0.5:
        raise ValueError("noise_density can't be higher than 0.5")
    max_input_tokens = length - 1
    min_input_tokens = min(
        max_input_tokens, max(1, round((1 - 2 * noise_density) * max_input_tokens))
    )
    num_input_tokens = torch.randint(min_input_tokens, max_input_tokens + 1, size=(1,))
    return torch.arange(length) > num_input_tokens


def noise_span_to_unique_sentinel(tokens, noise_mask, vocab_size):
    """implementation of t5.data.preprocessors.noise_span_to_unique_sentinel but in pytorch"""
    prev_token_is_noise = torch.nn.functional.pad(noise_mask[:-1], (1, 0))

    first_noise_tokens = torch.logical_and(
        noise_mask, torch.logical_not(prev_token_is_noise)
    )
    subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)

    sentinel = (vocab_size - 1) + 1 - torch.cumsum(first_noise_tokens, dim=0)

    tokens = torch.where(first_noise_tokens, sentinel, tokens)
    return torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))


def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocab_size):
    return noise_span_to_unique_sentinel(
        tokens, torch.logical_not(noise_mask), vocab_size
    )


def noise_span_to_fixed_sentinel(tokens, noise_mask, sentinel_value):
    prev_token_is_noise = torch.nn.functional.pad(noise_mask[:-1], (1, 0))

    first_noise_tokens = torch.logical_and(
        noise_mask, torch.logical_not(prev_token_is_noise)
    )
    subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)

    tokens = torch.where(first_noise_tokens, sentinel_value, tokens)
    return torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))


def nonnoise_span_to_fixed_sentinel(tokens, noise_mask, sentinel_value):
    return noise_span_to_fixed_sentinel(
        tokens, torch.logical_not(noise_mask), sentinel_value
    )


def random_spans_helper(
    inputs_length,
    noise_density,
    mean_noise_span_length,
    extra_tokens_per_span_inputs,
    extra_tokens_per_span_targets,
    decoder_only=False,
):
    """implementation of t5.data.preprocessors.random_spans_helper but in pytorch"""

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        return (
            num_nonnoise_tokens + num_noise_spans * extra_tokens_per_span_inputs,
            num_noise_tokens + num_noise_spans * extra_tokens_per_span_targets,
        )

    tokens_length = inputs_length
    if decoder_only:
        while (
            sum(_tokens_length_to_inputs_length_targets_length(tokens_length))
            > inputs_length
        ):
            tokens_length -= 1
    else:
        while (
            _tokens_length_to_inputs_length_targets_length(tokens_length)[0]
            <= inputs_length
        ):
            tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length
    )
    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def denoise(tokens, noise_mask, inputs_fn, targets_fn):
    """see t5.data.preprocessors.single_example_denoise for reference"""
    inputs = inputs_fn(tokens, noise_mask)
    targets = targets_fn(tokens, noise_mask)

    return inputs, targets
