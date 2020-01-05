import numpy as np


def vectorize(func):
    return np.vectorize(func)


@vectorize
def main_activity_choice(
        employment,
        density_work_places_mm,
        occupation,
        hidden,
        income_mm,
        hh_size,
        dist_closest_centres_mm,
        density_leisure_mm,
        age
):
    """
    0: home (ill, wfh)
    1: work
    2: education
    3: shopping
    4: leisure
    5: health
    :return:
    """
    if employment in [0, 1]:  # retired or unemployed
        p0 = 3
        p3 = (hh_size / 10) + dist_closest_centres_mm
        p4 = 1 + density_leisure_mm - (age / 100) + income_mm
        p5 = (age + hh_size) / 100
        choice = np.array([p0, p3, p4, p5])
        choice = choice / sum(choice)

        return np.random.choice([0, 3, 4, 5], p=choice)

    if employment == 2:  # in education
        if age > 16:
            p0 = .1  # ill or wfh
            p5 = .01
            p2 = 1 - p0 - p5
            return np.random.choice([0, 2, 5], p=[p0, p2, p5])

        p0 = .05  # ill or wfh
        p5 = .04
        p2 = 1 - p0 - p5
        return np.random.choice([0, 2, 5], p=[p0, p2, p5])

    if employment == 3:  # work
        p0 = .05 + (occupation * hidden * density_work_places_mm / 200)  # ill or wfh
        p5 = .01
        p1 = 1 - p0 - p5
        return np.random.choice([0, 1, 5], p=[p0, p1, p5])
