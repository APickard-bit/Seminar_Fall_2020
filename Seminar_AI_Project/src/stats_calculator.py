import math


def calc_linear_reg(stdev1, stdev2, mean1, mean2, coefficient1, isMultiVariable, set1=None, set2=None, set3=None,
                    mean3=None):
    if isMultiVariable is False:
        slope = coefficient1 * (stdev2 / stdev1)
        yIntercept = mean2 - (slope * mean1)

        return float(format(slope, '.2f')), float(format(yIntercept, '.2f'))

    else:

        sumSet1Sq = 0
        sumSet2Sq = 0
        sumSet2Set3 = 0
        sumSet1Set2 = 0
        sumSet1Set3 = 0

        for value in set1:
            sumSet1Sq += (value ** 2)

        for value in set2:
            sumSet2Sq += (value ** 2)

        for i in range(0, len(set1)):
            sumSet2Set3 += (set2[i] * set3[i])
            sumSet1Set2 += (set1[i] * set2[i])
            sumSet1Set3 += (set1[i] * set3[i])

        slope1 = ((sumSet2Sq * sumSet1Set3) - (sumSet1Set2 * sumSet2Set3)) / (
                    (sumSet1Sq * sumSet2Sq) - (sumSet1Set2 ** 2))
        slope2 = ((sumSet1Sq * sumSet2Set3) - (sumSet1Set2 * sumSet1Set3)) / (
                    (sumSet1Sq * sumSet2Sq) - (sumSet1Set2 ** 2))

        yIntercept = mean3 - slope1 * (mean1) - slope2 * (mean2)

        return float(format(slope1, '.2f')), float(format(slope2, '.2f')), float(format(yIntercept, '.2f'))


def calc_correlation_coefficiant(stdev1, stdev2, mean1, mean2, set1, set2):
    subset1 = []
    subset2 = []
    finalSum = 0
    for value in set1:
        subset1.append((value - mean1))

    for value in set2:
        subset2.append((value - mean2))

    for i in range(0, len(set1)):
        finalSum += subset1[i] * subset2[i]

    return float(format(finalSum / (len(set1) * stdev1 * stdev2), '.3f'))


def calc_std_dev(set, mean):
    sum = 0
    for value in set:
        sum += (value - mean) ** 2

    variance = sum / len(set)
    return float(format(math.sqrt(variance), '.3f'))


def calc_sample_mean(set):
    sum = 0

    for value in set:
        sum += value

    return float(format(sum / len(set), '.2f'))


def calc_sample_median(set):
    length = len(set)

    if length % 2 == 0:
        val1 = set[int((length - 1) / 2)]
        val2 = set[int(((length - 1) / 2) + 1)]
        return (val1 + val2) / 2
    else:
        index = (length / 2) + .5
        return set[int(index)]


def calc_sample_mode(set):
    counts = {}

    for value in set:
        if str(value) in counts:
            counts[str(value)] += 1
        else:
            counts[str(value)] = 1

    vals = counts.values()
    mode = max(vals)

    if mode > 1:
        return mode
    return 0
