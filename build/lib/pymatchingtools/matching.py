from pymatchingtools.functions import standardized_mean_difference, variance_ratio, EmpiricalCDF, ks_boot_test, average
from pymatchingtools.utils import NoSampleError, VariableError, VariableNoFoundError, SampleError

from pandas import DataFrame, Series, concat 

from numpy.random import normal
from numpy import mean, int64

from scipy.stats import ttest_ind

from statsmodels.genmod.generalized_linear_model import GLM as GLM

from lightgbm import \
    Dataset as lgbmDataset, \
    train as lgbmtrain

from statsmodels.api import families
from typing import List

from patsy import dmatrices

from matplotlib.pyplot import \
    legend as pltlegend, \
    xlim as pltxlim, \
    ylabel as pltylabel, \
    xlabel as pltxlabel, \
    title as plttitle, \
    subplot as pltsubplot, \
    figure as pltfigure, \
    show as pltshow, \
    savefig as pltsavefig

from seaborn import distplot as snsdistplot




class PropensityScoreMatch:
    def __init__(self, data: DataFrame) -> None:
        self.data = data
        self.outcome_dict = {}
        self.match_info = False
        self.fit_method = None
        self.is_fliter = True

    def balance_check(self, df:DataFrame, treatment_var: str, xvars: List[str], **kwargs) -> DataFrame:
        """
        Checking the balance of data source covariates
        
        Parameters
        ----------
        df: pd.DataFrame
            Data source
        treatment_var: str
            Variable name used to distinguish between treatment and control groups
        xvars: List[str]
            List of covariates
        smd_method: str (optional)
            The ways to calculate standardised_mean_difference. Default is 'cohen_d'. Support {'cohen_d', 'hedges_g', 'glass_delta'}
        smd_index_method: str (optional)
            One way of calculating, using either the mean or the median. Default is 'mean'. Support {'mean', 'median'}
        summary_print: str (optional)
            Whether to print the results of covariate balance checks


        Returns
        ----------
        summary_df: pd.Dataframe
            Results of covariate balance checks, including mean, SMD, VR ratio, Kolmogorov Smirnov Boost Test p-value
        """
        summary_df_dict = {}

        if kwargs.get('smd_method') is None:
            smd_method = 'cohen_d'
        else:
            smd_method = kwargs['smd_method']

        if kwargs.get('smd_index_method') is None:
            smd_index_method = 'mean'
        else:
            smd_index_method = kwargs['smd_index_method']

        xvars = list(filter(lambda x: x not in ['Intercept'], xvars))

        df[treatment_var] = df[treatment_var].astype(int64)

        for xvar in xvars:
            # print(treatment_var, xvar)
            # print(df.loc[lambda x: x[treatment_var] == 0.0, :])
            try:
                control = df.loc[lambda x: x[treatment_var] == 0, xvar]
                treatment = df.loc[lambda x: x[treatment_var] == 1, xvar]
            except:
                raise NoSampleError(var=xvar)

                

            mean_treated = mean(treatment)
            mean_control = mean(control)

            

            smd = standardized_mean_difference(
                control, 
                treatment, 
                method=smd_method, 
                index_method=smd_index_method
            )

            vr = variance_ratio(control=control, treatment=treatment)

            if kwargs.get('n_boots') is None:
                n_boots = 1000
            else:
                n_boots = kwargs('n_boots')

            if kwargs.get('alternative') is None:
                alternative = 'two_sided'
            else:
                alternative = kwargs['alternative']

            if kwargs.get('eps') is None:
                eps = 2**-53
            else:
                eps = kwargs['eps']

            ks_pvalue = ks_boot_test(
                control=control,
                treatment=treatment,
                n_boots=n_boots,
                alternative=alternative, 
                eps=eps
            )

            summary_df_dict[xvar] = {
                'Means Treated': mean_treated,
                'Means Control': mean_control,
                'Std. Mean Diff.': smd,
                'Var. Ratio': vr,
                'ks p-value': ks_pvalue,
            }
        
        summary_df = DataFrame(summary_df_dict).transpose()
        
        if kwargs.get('summary_print_str') is not None:
            summary_print_str = kwargs.get('summary_print_str') + ' ' + 'balance check result'
        else:
            summary_print_str = 'balance check result'

        if kwargs.get('summary_print', False) == True:
            
            print('-'*10, summary_print_str, '-'*10)
            print(summary_df)
            print(f'control size={len(control)}, treatment size = {len(treatment)}')

        return summary_df


    def get_match_info(self, formula: str=None, xvars: list=None, yvar: str=None, categorical_feature_list: list=None, save: bool=True, **kwargs) -> DataFrame:
        """
        Get the variables information
        
        Parameters
        ----------
        formula: str (optional)
            A custom formula, patsy format. Need to specify variable formula variable, or the variable combination x and y
        xvars: List[str] (optional)
            List of covariates. Need to specify variable formula variable, or the variable combination x and y
        yvar: str
            Variable name used to distinguish between treatment and control groups. Need to specify variable formula variable, or the variable combination x and y
        categorical_feature_list: list (optional)
            Dummy variable
        smd_method: str (optional)
            The ways to calculate standardised_mean_difference. Default is 'cohen_d'. Support {'cohen_d', 'hedges_g', 'glass_delta'}
        smd_index_method: str (optional)
            One way of calculating, using either the mean or the median. Default is 'mean'. Support {'mean', 'median'}
        summary_print: str (optional)
            Whether to print the results of covariate balance checks


        Returns
        ----------
        summary_df: pd.Dataframe
            Results of covariate balance checks, including mean, SMD, VR ratio, Kolmogorov Smirnov Boost Test p-value
        """


        if kwargs.get('smd_method') is None:
            smd_method = 'cohen_d'
        else:
            smd_method = kwargs['smd_method']

        if kwargs.get('smd_index_method') is None:
            smd_index_method = 'mean'
        else:
            smd_index_method = kwargs['smd_index_method']
        
        if formula is not None:
            y, x = dmatrices(formula_like=formula, data=self.data, return_type='dataframe')

            yvar = y.columns[0]
            xvars = x.columns
        elif xvars is not None and yvar is not None:
            xvars_list = []
            if set(categorical_feature_list) - set(xvars) == set() :
                xvars_list = [f'C({categorical_feature}, Treatment)' for categorical_feature in categorical_feature_list] + [f'C({feature}, Treatment)' for feature in set(xvars) - set(categorical_feature_list)]
            else:
                raise VariableNoFoundError(set(categorical_feature_list) - set(xvars))

            x_var_format = '+'.join(xvars_list)
            
            formula = f'{yvar} ~ {x_var_format}'

            y, x = dmatrices(formula_like=formula, data=self.data, return_type='dataframe')

            yvar = y.columns[0]
            xvars = x.columns
        else:
            raise VariableError('You need to give the formula, or specify the variables x and y.')

        try:
            if {int(x[0]) for x in y.value_counts().index.values} == {0, 1}:
                pass
        except:
            raise VariableError('Only binary variables can be accepted, and now only 0 and 1 are supported for variable values.')

        if save:
            self.y: DataFrame = y
            self.x: DataFrame = x
            self.yvar = yvar
            self.xvars = xvars

        summary_print = kwargs.get('summary_print', False)
        
        summary_df = self.balance_check(
            df=concat([x, y], axis=1),
            treatment_var=yvar,
            xvars=list(xvars),
            smd_method=smd_method,
            smd_index_method=smd_index_method,
            summary_print=summary_print
        )

        return summary_df

    def predict_score(self, model: any, method: str, x: DataFrame) -> Series:
        """
        Predict propensity scores
        
        Parameters
        ----------
        model: any
            Model of predicted propensity scores
        method: str
            Name of the propensity scores predict model. Default is 'glm'. Support {'glm', 'lgbm'}
        x: pd.Dataframe
            Data representing for covariates

        Returns
        ----------
        scores: pd.Series
            Predicted propensity scores
        """
        if method == 'glm':
            scores = model.predict(x)
        elif method == 'lgbm':
            scores = model.predict(x)
        
        return scores

    def plot_scores(self, control_scores: Series, treatment_scores: Series, title_name: str=None, savefig: bool= False, savepath: str=None) -> None:
        """
        Plot the distribution of propensity scores
        
        Parameters
        ----------
        control_scores: Series
            predicted propensity scores of control group
        treatment_scores: Series
            predicted propensity scores of treatment group
        """
        pltfigure(dpi=300)

        snsdistplot(control_scores, label='Control')
        snsdistplot(treatment_scores, label='Treatment')

        pltxlim((0, 1))

        if title_name is None:
            title_name = 'Propensity Scores Distribution'

        
        plttitle(title_name)
        pltxlabel('Propensity Scores')
        pltylabel('Percentage')

        if savefig == True:
            if savepath is None:
                savepath = 'Default.png'
            pltsavefig(savepath)

        pltshow()
        
    
    def fit(self, x: DataFrame, y: DataFrame, data: DataFrame, method='glm', save: bool=True, fit_print=False, **kwargs) -> tuple:
        """
        Predict propensity scores
        
        Parameters
        ----------
        
        x: pd.Dataframe
            Data representing for covariates
        y: pd.Dataframe
            Variable  used to distinguish between treatment and control groups.
        data: pd.DataFrame
            Data source
        method: str (optional)
            Name of the propensity scores predict model. Default is 'glm'. Support {'glm', 'lgbm'}

        Returns
        ----------
        scores: pd.Series
            Predicted propensity scores
        """


        self.fit_method = method

        if method == 'glm':
            glm = GLM(y, x, family=families.Binomial())
            model = glm.fit()
            if fit_print == True:
                print(model.summary())
        elif method == 'lgbm':
            if kwargs.get('params') is not None:
                params = kwargs.get('params')
            else:
                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'binary_logloss', 
                    'num_leaves': 32,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.5,
                    'bagging_freq': 5,
                    'verbose': 0,
                }

            train_set = lgbmDataset(x, y)
            model = lgbmtrain(
                params=params,
                train_set=train_set,
                num_boost_round=100
            )

            # model = lgbm.LGBMClassifier()
            # model.fit(X=self.x, y=self.y)
            
        
        scores = self.predict_score(model=model, method=method, x=x)
        data['scores'] = scores
        

        if save:
            self.model = model
            self.data = data

        return model, data

    
    def match(
        self, 
        threshold: float=0.01, 
        distance: str='manhattan', 
        fit_method: str= 'glm',
        method: str='min', 
        k=1, 
        save: bool=True, 
        data: DataFrame=None, 
        is_fliter = True,
        **kwargs
    ) -> DataFrame:
        """
        Propensity score matching
        
        Parameters
        ----------
        threshold : float (optional)
            Threshold for fuzzy matching matching. 
            i.e. for manhattan distance : |score_x - score_y| >= theshold
        distance: str (optional)
            Measures of propensity score distance that currently only supports {'Manhattan'}
        fit_method: str (optional)
            Name of the propensity scores predict model. Default is 'glm'. Support {'glm', 'lgbm'}
        method: str (optional)
            Method of sample selection. Currently only supports {'min'}
        k: int (optional)
            Number of sample selection. Default is 1
        is_filter: bool (optional)
            is_filter=True means pull-back sampling. is_filter=False means random sample without putback (i.e. without prior sampling). Default is True


        Returns
        ----------
        matched_data: pd.Dataframe
            Data after propensity score matching
        """
        # self.threshold = threshold
        # self.distance = distance
        # self.match_method = method
        # self.k = k
        self.is_fliter = is_fliter

        if data is None:
            data = self.data

        if 'scores' not in data.columns:
            self.fit(x=self.x, y=self.y, data=data, method=fit_method, save=False)
        if distance == 'manhattan':
            treatment_scores: Series = data.loc[lambda x: x[self.yvar]==1, 'scores']
            control_scores: Series = data.loc[lambda x: x[self.yvar]==0, 'scores']

            results, match_ids = [], []

            for i in range(len(treatment_scores)):
                match_id = i
                score = treatment_scores.iloc[i]
                

                if method == 'min':
                    matches: Series = (abs(control_scores - score) <= threshold).sort_values().head(k)
                    matches_index = matches.index

                results.extend([treatment_scores.index[match_id]] + list(matches_index))
                match_ids.extend([match_id]*(len(matches_index)+1))

                if is_fliter == True:
                    try:
                        control_scores = control_scores.drop(index=matches.index)
                    except:
                        raise SampleError('Insufficient sample size to find samples to exclude')

            matched_data = data.loc[results]
            matched_data['match_id'] = match_ids
            matched_data['record_id'] = matched_data.index
            
        if save:
            self.matched_data = matched_data
            

        return matched_data
            
    def ATE(self, data: DataFrame, treatment_var: str, outcome_var: str, **kwargs) -> None:
        """
        Checking the balance of data source covariates
        
        Parameters
        ----------
        data: pd.DataFrame
            Data source
        treatment_var: str
            Variable name used to distinguish between treatment and control groups
        outcome_var: str
            Implicit variable
        """

        if kwargs.get('outcome_dict') is not None and kwargs.get('outcome_name') is not None:
            outcome_dict = kwargs.get('outcome_dict')
            outcome_name = kwargs.get('outcome_name')

        data_group = data.groupby(by=[treatment_var]).agg(
            {
                outcome_var: ['count', 'sum', average, 'var',]
            }
        )
        data_group = data_group[outcome_var]
        data_group = data_group.reset_index()

        res = ttest_ind(
            data.loc[lambda x: x[treatment_var]==1, outcome_var],
            data.loc[lambda x: x[treatment_var]==0, outcome_var]
        )

        # ate = data.loc[lambda x: x[treatment_var]==1, outcome_var] - data.loc[lambda x: x[treatment_var]==0, outcome_var]
        ate = data_group.loc[lambda x: x[treatment_var]==1, 'average'].values - data_group.loc[lambda x: x[treatment_var]==0, 'average'].values

        
        t_stats = res.__getattribute__('statistic')
        p_value = res.__getattribute__('pvalue')

        if kwargs.get('outcome_print_str') is None:
            print_result_str = 'result'
            print_stats_str = 'stats'
        else:
            print_result_str = kwargs.get('outcome_print_str') + ' ' +'result'
            print_stats_str = kwargs.get('outcome_print_str') + ' ' +'stats'
        
        print('-'*10, print_result_str, '-'*10)
        print(data_group)

        print('-'*10, print_stats_str, '-'*10)
        print(f'ATE = {ate}, stats = {t_stats}, p-value = {p_value}')

        if kwargs.get('outcome_dict') is not None and kwargs.get('outcome_name') is not None:
            outcome_dict = kwargs.get('outcome_dict')
            outcome_name = kwargs.get('outcome_name')

            outcome_dict[outcome_name] = {
                'ate': ate,
                't-stats': t_stats,
                'p-value': p_value,
            }
        
        return


    def __randomized_confounding_test(
        self, 
        outcome_var: str, 
        threshold=0.01, 
        distance: str='manhattan', 
        match_method: str='min', 
        k=1,
        is_fliter=True
    ):
        random_var = normal(loc=0.0, scale=1.0, size=len(self.x))

        data_new = self.data.copy(deep=True)
        data_new['random_var'] = random_var

        y = self.y
        yvar = self.yvar

        x_new = self.x.copy(deep=True)
        x_new['random_var'] = random_var


        _, data_new = self.fit(x=x_new, y=y, data=data_new, method=self.fit_method, save=False)

        matched_data_new = self.match(
            threshold=threshold,
            distance=distance,
            method=match_method,
            k=k,
            save=False,
            data=data_new,
            is_fliter=is_fliter
        )

        outcome_str = 'randomized_confounding_test'
        self.ATE(
            data=matched_data_new, 
            treatment_var=yvar, 
            outcome_var=outcome_var,
            outcome_dict = self.outcome_dict,
            outcome_name = outcome_str,
            outcome_print_str = outcome_str
        )

        return

        
        

    def __placebo_test(
        self,
        outcome_var: str, 
    ):
        outcome_var_new = normal(loc=0, scale=1, size=len(self.matched_data))
        
        yvar = self.yvar

        matched_data_new = self.matched_data.copy(deep=True)
        matched_data_new[outcome_var] = outcome_var_new

        outcome_str = 'placebo_test'
        self.ATE(
            data=matched_data_new, 
            treatment_var=yvar, 
            outcome_var=outcome_var,
            outcome_dict = self.outcome_dict,
            outcome_name = outcome_str,
            outcome_print_str = outcome_str
        )

        return
    
    def __subset_data_test(
        self, 
        outcome_var: str, 
        threshold=0.01, 
        distance: str='manhattan', 
        match_method: str='min', 
        k=1,
        frac: float=0.5,
        is_fliter=True
    ):
        sample_df = self.data.sample(frac=frac).copy(deep=True)

        y = self.y
        yvar = self.yvar
        x = self.x.copy(deep=True)

        _, data_new = self.fit(x=x, y=y, data=sample_df, method=self.fit_method, save=False)

        matched_data_new = self.match(
            threshold=threshold,
            distance=distance,
            method=match_method,
            k=k,
            save=False,
            data=data_new,
            is_fliter=is_fliter
        )
        outcome_str = 'subset_data_test'
        self.ATE(
            data=matched_data_new, 
            treatment_var=yvar, 
            outcome_var=outcome_var,
            outcome_dict = self.outcome_dict,
            outcome_name = outcome_str,
            outcome_print_str = outcome_str
        )

        return

    def __gamma(self):
        pass
    
    # @property
    # def matched_data(self):
    #     return self.matched_data

    # @self.setter
    # def matched_data(self, data_new):
    #     self.matched_data = data_new

    def after_match_check(
        self, 
        outcome_var: str, 
        frac: float=0.5
    ):
        """
        Post-matching covariate balance checks, and corresponding refutation tests and sensitivity analyses
        
        Parameters
        ----------
        outcome_var: str
            Implicit variable
        frac: float (optional)
            Sample rate for subset data test from 0 and 1. Default is 0.5
        """
         
        # matched_data = self.match(
        #     threshold=threshold,
        #     distance=distance,
        #     method=match_method,
        #     k=k,
        #     save=True,
        #     data=self.data
        # )
        is_fliter = self.is_fliter

        matched_data = self.matched_data

        self.ATE(
            data=matched_data, 
            treatment_var=self.yvar, 
            outcome_var=outcome_var,
        )
        
        self.balance_check(df=matched_data, treatment_var=self.yvar, xvars=self.xvars, summary_print=True, summary_print_str='after matched')
        
        self.__randomized_confounding_test(outcome_var=outcome_var, is_fliter=is_fliter)
        self.__placebo_test(outcome_var)
        
        self.__subset_data_test(outcome_var=outcome_var, frac=frac, is_fliter=is_fliter)
        self.__gamma()




        
            

# if __name__ == '__main__':
#     column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#     data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

    
#     formula = 'CHAS ~ CRIM + ZN + INDUS + NOX + RM + AGE + DIS + RAD'

#     matcher = Matching(data=data)
#     summary_df = matcher.get_match_info(formula=formula, summary_print=True)
#     # model = matcher.fit(method='glm')

#     matched_data = matcher.match(
#         method='min',
#         is_fliter=True
#     )

#     matcher.after_match_check(
#         outcome_var='MEDV',
#         frac=0.8,
#         match_method='min'
#     )



