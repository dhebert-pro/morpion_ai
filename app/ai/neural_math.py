import math


def sigmoid(value):
    if value < -60:
        return 0.0

    if value > 60:
        return 1.0

    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_derivative_from_output(output):
    return output * (1.0 - output)


def tanh_derivative_from_output(output):
    return 1.0 - output * output


def compute_masked_mean_squared_error(predictions, targets, legal_moves_mask):
    total_error = 0.0
    used_outputs_count = 0

    for index in range(len(predictions)):
        if legal_moves_mask[index] == 0:
            continue

        error = targets[index] - predictions[index]
        total_error += error * error
        used_outputs_count += 1

    if used_outputs_count == 0:
        return 0.0

    return total_error / used_outputs_count
