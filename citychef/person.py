import numpy as np


def vectorize(func):
    return np.vectorize(func)


@vectorize
def get_is_adult(agent_hh_index, hh_count, hh_children):
    adults = hh_count - hh_children
    if agent_hh_index < adults:
        return 1
    else:
        return 0


@vectorize
def get_gender(hh_agent_index, agent_is_adult, hh_children, agent_hidden):

    # male: 0
    # female: 1
    # other: 2

    if not agent_is_adult:
        return np.random.choice([0, 1, 2], p=[.47, .51, .02])

    elif hh_children:  # number of children
        p = agent_hidden / 10  # hidden hh
        po = agent_hidden / 100

        if hh_agent_index % 2:
            return np.random.choice([0, 1, 2], p=[1-p-po, p, po])
        else:
            return np.random.choice([0, 1, 2], p=[p, 1-p-po, po])
    else:
        pm = .5 + agent_hidden / 20  # hidden hh
        po = agent_hidden / 200
        return np.random.choice([0, 1, 2], p=[pm, 1-pm-po, po])


@vectorize
def get_age(hh_hidden, hh_age_group, adult):

    if not adult:
        return int(np.random.uniform(0, 11) + np.random.triangular(0, hh_hidden, 6))

    return hh_age_group + int(np.random.poisson(2))


@vectorize
def employment(adult, hh_people_in_work, p_hh_index, age, hidden, dist_education_mm, dist_work_mm, hh_income_mm, density_mm):
    """
    0: retired
    1: unemployed
    2: education
    3: employed
    """
    if not adult:
        return 2

    if p_hh_index < hh_people_in_work:

        if age < 20:
            peducation = hidden / 10 + dist_education_mm + hh_income_mm / hh_people_in_work
            pwork = 5 + dist_work_mm + density_mm
            choice = np.array([peducation, pwork])
            choice = choice / sum(choice)
            return np.random.choice([2, 3], p=choice)

        peducation = 1 - hidden / 10
        pwork = 10 + dist_work_mm + density_mm + (age - 20)
        choice = np.array([peducation, pwork])
        choice = choice / sum(choice)
        return np.random.choice([2, 3], p=choice)

    else:

        if age > 73:
            return 0

        if age > 50:
            pretired = 1 + hh_income_mm - density_mm + (age - 55) / 5
            punemployed = density_mm + dist_education_mm
            pwork = 1 + hidden
            choice = np.array([pretired, punemployed, pwork])
            choice = choice / sum(choice)
            return np.random.choice([0, 1, 3], p=choice)

        punemployed = hidden/10 + density_mm + dist_education_mm
        pwork = 1 + hidden/10 + dist_work_mm
        choice = np.array([punemployed, pwork])
        choice = choice / sum(choice)
        return np.random.choice([1, 3], p=choice)


@vectorize
def occupation(employed, age, income_mm):
    """
    0:None
    1:Unskilled
    2:Skilled
    3:Manager
    """
    if not employed == 3:
        return 0

    p = age * income_mm / 100

    return 1 + np.random.binomial(2, p, size=None)
