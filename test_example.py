import pandas as pd
from pymatchingtools.matching import PropensityScoreMatch


if __name__ == '__main__':
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

    
    formula = 'CHAS ~ CRIM + ZN + INDUS + NOX + RM + AGE + DIS + RAD'

    matcher = PropensityScoreMatch(data=data)
    summary_df = matcher.get_match_info(formula=formula, summary_print=True)

    matched_data = matcher.match(
        method='min',
        is_fliter=True,
        fit_method='glm'
    )

    matcher.after_match_check(
        outcome_var='MEDV',
        frac=0.8,
    )