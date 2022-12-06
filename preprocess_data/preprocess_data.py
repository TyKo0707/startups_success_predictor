import pandas as pd
import pycountry_convert as pc


def country_to_continent(country_name):
    # This function decides which continent does the country belong to
    country_alpha2 = pc.country_alpha3_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name


df3 = pd.read_csv('../datasets/companies.csv')

# Dropping useless columns
df3.drop(['homepage_url', 'state_code', 'region', 'city', 'first_funding_at', 'last_funding_at'], inplace=True, axis=1)

# Dropping rows with NaN values in chosen columns
df3.dropna(subset=['founded_at', 'funding_total_usd', 'category_list', 'country_code'], inplace=True)
df3 = df3[df3.funding_total_usd != '-']

# Indexing our data
s = [i for i in range(1, df3.shape[0] + 1)]
df3['permalink'] = s
df3.rename(columns={'permalink': 'company_index'}, inplace=True)

# Replacing status data with numbers
df3.status = df3.status.replace(['closed'], 0)
df3.status = df3.status.replace(['operating', 'acquired'], 1)
df3.status = df3.status.replace(['ipo'], 2)

df3.funding_total_usd = df3.funding_total_usd.astype('float64').apply(int)

# Change incorrect country codes to correct
df3.country_code = df3.country_code.replace(['ROM'], 'ROU')
df3.country_code = df3.country_code.replace(['BAH'], 'BHS')
df3.country_code = df3.country_code.replace(['TAN'], 'TZA')

# Change country_code column to continent which does the country belong to
list_of_continents = [country_to_continent(i) for i in df3.country_code]
df3.country_code = list_of_continents
df3.rename(columns={'country_code': 'region'}, inplace=True)

# get the description (count, mean, standard deviation, min and max value, quartiles) of numerical data
print(df3.describe().transpose())

# get the description (count, unique, top, freq) of categorical data
print(df3.describe(include=['O']).transpose())

# Saving out new dataset
df3.to_csv('../main_dataset.csv', index=False)
