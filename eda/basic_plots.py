import pandas as pd
from environs import Env
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

env = Env()
env.read_env()
EXT_MAIN_DATASET_PATH = env.str("EXT_MAIN_DATASET_PATH")
MAIN_DATASET_PATH = env.str("MAIN_DATASET_PATH")
PLOTS_PATH = env.str('PLOTS_PATH')

df = pd.read_csv(EXT_MAIN_DATASET_PATH)

sns.set_theme(style="darkgrid", palette='tab10')

regions = df.groupby(['region']).size().index.values

# region Companies by region (bar plot)
comp_by_regions = df.groupby(['region']).size().values
idx = np.argsort(comp_by_regions)[::-1]

comp_by_regions = np.array(comp_by_regions)[idx]
regions_1 = np.array(regions)[idx]
plt.figure(figsize=(15, 8))
sns.barplot(x=regions_1, y=comp_by_regions).set(title='Number of companies by region', xlabel='Regions',
                                                ylabel='Number of companies')
plt.savefig(PLOTS_PATH + "comp_by_regions.png")
plt.show()
# endregion

# region Mean funding_total_usd by regions (bar plot)
fund_by_regions = df.groupby(['region']).funding_total_usd.mean()
idx = np.argsort(fund_by_regions)[::-1]

fund_by_regions = np.array(fund_by_regions)[idx]
regions_2 = np.array(regions)[idx]
plt.figure(figsize=(15, 8))
sns.barplot(x=regions_2, y=fund_by_regions).set(title='Mean value of total funding by company by regions',
                                                xlabel='Regions',
                                                ylabel='Total funding by company')
plt.savefig(PLOTS_PATH + "fund_by_regions.png")
plt.show()
# endregion

# region Mean funding_total_usd by company by year (lineplot)
df['year'] = df.founded_at.str.slice(stop=4).astype(int)
df = df[df.year >= 2000]

mean_by_year = df.groupby(['region', 'year']).funding_total_usd.mean()
mean_by_year = mean_by_year.reset_index()
mean_by_year = mean_by_year.drop(mean_by_year[mean_by_year.year > 2015].index)
plt.figure(figsize=(15, 8))
sns.lineplot(x='year',
             y='funding_total_usd',
             hue='region',
             data=mean_by_year).set(title='Mean value of total funding by company by regions (by year)',
                                    xlabel='Regions',
                                    ylabel='Total funding by company')
plt.savefig(PLOTS_PATH + "mean_fund_by_year.png")
plt.show()
# endregion

# region Distribution by categories (pieplot)
df.category_list = df.category_list.str.split('|')
cl = []
for i in df.category_list.values:
    for j in i:
        cl.append(j)
categories, values = pd.Series(cl).value_counts()[:10].index.values, pd.Series(cl).value_counts()[:10]
values_perc = values / len(cl) * 100

plt.figure(figsize=(15, 8))
sns.barplot(x=categories, y=values_perc).set(
    title=f'Number of companies by category \n(the presented categories are included in '
          f'{round(100 - (len(cl) - values.sum()) * 100 / len(cl), 2)}% companies)',
    xlabel='Category',
    ylabel='% relative to the entire dataset')
plt.savefig(PLOTS_PATH + "category_distribution.png")
plt.show()
# endregion

# region The company in which the most investments were made (bar plot)
df1 = pd.read_csv(MAIN_DATASET_PATH)
data = df1.sort_values(by=['funding_total_usd'], ascending=True)

y = np.flipud(data.name.values[-20:])
x = np.flipud(data.funding_total_usd.values[-20:])
plt.figure(figsize=(18, 8))
sns.barplot(x=x, y=y).set(title='The company in which the most investments were made',
                          xlabel='Investment size',
                          ylabel='Company name')
sns.despine(left=True, bottom=True)
plt.savefig(PLOTS_PATH + "investments_by_company.png")
plt.show()
# endregion
