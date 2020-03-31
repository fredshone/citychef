import numpy as np


def minmax(array, axis=0):
    if len(array.shape) > 1:
        return (array - array.min(axis=axis)) / (array.max(axis=axis) - array.min(axis=axis))
    else:
        return (array - min(array)) / (max(array) - min(array))


def vectorize(func):
    return np.vectorize(func)


@vectorize
def gen_hidden(xx, yy, hh_density_mm):
    """
    hidden household variable, either [1, 2, 3, 4, 5]
    """
    p1 = np.random.poisson(xx*10)
    p2 = np.random.poisson(hh_density_mm*10)
    p3 = np.random.poisson(hh_density_mm*xx*10)
    p4 = np.random.poisson(((xx - .5)*10)**2)
    p5 = np.random.poisson(yy*10)
    choice = np.array([p1, p2, p3, p4, p5])
    choice = choice/sum(choice)
    return np.random.choice([1, 2, 3, 4, 5], p=choice)


@vectorize
def gen_hh_count(hidden, density):
    """
    num people in household, either [1, 2, 3, 4, 5]
    """
    p1 = density*50 + hidden*2
    p2 = density*40 + hidden*4
    p3 = density*30 + hidden*6
    p4 = density*20 + hidden*8
    p5 = density*10 + hidden*10
    choice = np.array([p1, p2, p3, p4, p5])
    choice = choice/sum(choice)
    return np.random.choice([1, 2, 3, 4, 5], p=choice)


@vectorize
def gen_num_children(count, hidden, hh_dist_closest_schools_mm, hh_density_leisure_mm):
    """
    1 person = a
    2 persons = aa, ac
    3 persons = aaa, aac, acc
    4 people = aaaa, aaac, aacc, accc
    5 people = aaaaa, aaaac, aaacc, aaccc, acccc

    """

    if count == 1:
        return 0

    else:
        pchild = hh_density_leisure_mm + hh_dist_closest_schools_mm + count / 2
        padult = 3 + hidden
        p = pchild / (pchild + padult)
        return np.random.binomial((count - 1), p, size=None)


@vectorize
def gen_age_group(hh_children, hh_hidden, hh_density_mm):

    if hh_children:
        return int(np.random.triangular(16, 16 + hh_hidden, 30 + 4 * hh_hidden))

    old = 20 + (2 * hh_hidden)
    adult = 10 + (2 * hh_hidden)
    p_old = 1 - hh_density_mm/2
    return np.random.choice(
        [
            30 + int(np.random.poisson(old)),
            20 + int(np.random.poisson(adult)),
        ],
        p=(p_old, 1-p_old)
    )


@vectorize
def get_people_in_work(hh_age_group, hh_hidden, hh_count, hh_num_children, hh_density_mm):

    if hh_age_group > 80:
        return 0

    adults = hh_count - hh_num_children

    if hh_hidden > 2:
        p = 2 + hh_hidden + hh_density_mm + np.random.poisson(1)
        pn = 2 + hh_num_children
    else:
        p = 2 + hh_hidden - hh_density_mm
        pn = 2 + hh_num_children + hh_density_mm + np.random.poisson(1)

    p *= (80 - hh_age_group)
    pn *= hh_age_group

    p = p / (1 + p + pn)

    return np.random.binomial(adults, p)


@vectorize
def get_in_work(people_in_work):

    if people_in_work:
        return 1
    else:
        return 0


@vectorize
def get_income(hh_in_work, hh_count, hh_num_children, hh_density_work_places_mm, hh_density_mm):

    if not hh_in_work:
        return (15 + hh_count + hh_num_children) * 1000

    adults = hh_count - hh_num_children
    pwork = (hh_density_work_places_mm + hh_density_mm) / 2
    working = 1 + np.random.binomial(adults - 1, pwork)

    income = 0
    for p in range(working):
        income += (15 + np.random.poisson(5*(hh_density_work_places_mm + 1 - hh_density_mm))*5) * 1000

    return income


@vectorize
def get_cars(hh_hidden, income_mm, density_mm, hh_count, hh_children):

    adults = hh_count - hh_children
    p1 = income_mm / 2 + (1 - density_mm) / 2
    if hh_hidden < 3:
        p1 /= 1.2
    p2 = p1 / 1.5
    return np.random.binomial(1, p1) + np.random.binomial(adults-1, p2)
