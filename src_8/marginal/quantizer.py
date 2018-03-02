import math
import numpy as np
from operator import itemgetter


class Quantizer:
    def __init__(self, x_min: float, x_max: float):
        self.__x_min = x_max/10

        self.__x_max = x_max
        self.__N = x_max

    def quantize(self, x: float) -> float:
        if x <= self.__x_min:
            return x
        return round(self.__x_max * math.log(x / self.__x_min) / math.log(self.__x_max / self.__x_min))


class Representative:
    # @quantized_values: list of pair of (x, y)
    def __init__(self, marg_model, domain_lower=0, domain_upper=1, span=0.001):
        xl = np.arange(domain_lower, domain_upper, span)
        yl = [marg_model.pdf(x) for x in xl]
        q = Quantizer(min(yl), max(yl))
        zl = [q.quantize(y) for y in yl]

        '''
        plt.plot(xl, yl)
        plt.plot(xl, zl)
        plt.show()
        '''

        l = []
        # Add a sentinel to head of list
        l.append(([domain_lower-1, None], None))
        for x, y in zip(xl, zl):
            last_elm = l[-1]
            if not y == last_elm[1]:
                l.append(([x, None], y))
                last_elm[0][1] = (last_elm[0][0] + x) / 2
        # Add a sentinel to tail of list
        last_elm = l[-1]
        last_elm[0][1] = (last_elm[0][0] + domain_upper) / 2
        l.append(([domain_upper + 1, None], None))
        self.__representative_list = [x[0] for x in l]

    def get_representative(self, x):
        l = self.__representative_list
        left = 0
        right = len(l) - 1
        while True:
            mid = (left + right) // 2
            if x < l[mid][0]:
                right = mid - 1
            elif x >= l[mid][0]:
                if x < l[mid + 1][0]:
                    return l[mid][1]
                else:
                    left = mid + 1


def get_peaks(xl, yl):
    peak_index_list = []
    length = len(yl)
    for i in range(1, length-1):
        if yl[i] > yl[i+1] and yl[i] > yl[i-1]:
            peak_index_list.append(i)
    return peak_index_list


def get_quantization_area(peak_i, xl, yl, peak_value, domain_lower=0, domain_upper=1, span=0.001):
    # go left
    peak = yl[peak_i]
    threshold = peak * peak_value
    left_i = search_i(peak_i, threshold, xl, yl, -1, domain_lower, domain_upper)
    right_i = search_i(peak_i, threshold, xl, yl, 1, domain_lower, domain_upper)
    return (xl[left_i], xl[right_i])


def search_i(peak_i, target, xl, yl, span, lower, upper):
    i = peak_i
    while True:
        i += span
        if i == 0 or i == 999:
            return i
        if yl[i] <= target:
            return i


class SimpleRepresentative:
    def __init__(self, marg_model, peak_value, domain_lower=0, domain_upper=1, span=0.001):
        xl = np.arange(domain_lower, domain_upper, span)
        yl = [marg_model.pdf(x) for x in xl]
        # get peak
        peak_index_list = get_peaks(xl, yl)
        q_area_list = []
        # get q area
        for peak_i in peak_index_list:
            q_area_list.append(get_quantization_area(peak_i, xl, yl, peak_value))
        # marge q area
        wrong_set = set()
        length = len(q_area_list)
        for i in range(length - 1):
            current = q_area_list[i]
            for j in range(i + 1, length):
                target = q_area_list[j]
                if (current[0] > target[0]) and (target[1] < current[1]):
                    wrong_set.add(current)
                elif (target[0] > current[0]) and (current[1] < target[1]):
                    wrong_set.add(target)
        area_set = set(q_area_list)
        for wrong in wrong_set:
            area_set.remove(wrong)
        self.__q_areas = sorted(list(area_set), key=itemgetter(0))

    def get_representative(self, x):
        q_areas = self.__q_areas
        for area in q_areas:
            if (area[0] <= x) and (x <= area[1]):
                return area[0] / area[1]
        return x


class NoRepresentative:
    def get_representative(self, x):
        return x
