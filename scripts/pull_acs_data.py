import os


import pandas as pd
from census import Census
from us import states

# Census API key
API_KEY = os.environ.get("CENSUS_API_KEY")

# ACS 2019 5-year
c = Census(API_KEY, year=2019)

GA_FIPS = states.GA.fips

# Atlanta metro counties
# Henry, Rockdale, Newton, Coweta, Forsyth, Paulding, Spalding, Walton
ATLANTA_COUNTIES = [
    "067",  # Cobb
    "089",  # DeKalb
    "121",  # Fulton
    "135",  # Gwinnett
    "063",  # Clayton
    "151",  # Henry
    "223",  # Paulding
]


# ============================================================
# 1. Housing stock by structure type (B25024)
# ============================================================

housing_stock = []

for county in ATLANTA_COUNTIES:
    data = c.acs5.state_county_tract(
        fields=(
            "NAME",
            "B25024_001E",
            "B25024_002E",  # detached SFR
            "B25024_003E",  # attached SFR/townhouse
            "B25024_004E",  # 2 units
            "B25024_005E",  # 3-4 units
            "B25024_006E",  # 5-9 units
            "B25024_007E",  # 10-19 units
            "B25024_008E",  # 20-49 units
            "B25024_009E",  # 50+ units
        ),
        state_fips=GA_FIPS,
        county_fips=county,
        tract=Census.ALL,
    )

    housing_stock.extend(data)


df_stock = pd.DataFrame(housing_stock)

df_stock.rename(
    columns={
        "B25024_001E": "total_units",
        "B25024_002E": "sfr_detached",
        "B25024_003E": "sfr_attached",
        "B25024_004E": "two_unit",
        "B25024_005E": "small_multifamily",
        "B25024_006E": "mf_5_9",
        "B25024_007E": "mf_10_19",
        "B25024_008E": "mf_20_49",
        "B25024_009E": "mf_50plus",
    },
    inplace=True,
)


df_stock.to_csv("atlanta_housing_stock_2019.csv", index=False)


# ============================================================
# 2. Housing tenure (B25003)
# ============================================================

tenure = []

for county in ATLANTA_COUNTIES:
    data = c.acs5.state_county_tract(
        fields=(
            "NAME",
            "B25003_001E",
            "B25003_002E",
            "B25003_003E",
        ),
        state_fips=GA_FIPS,
        county_fips=county,
        tract=Census.ALL,
    )

    tenure.extend(data)


df_tenure = pd.DataFrame(tenure)

df_tenure.rename(
    columns={
        "B25003_001E": "total_occupied",
        "B25003_002E": "owner_occupied",
        "B25003_003E": "renter_occupied",
    },
    inplace=True,
)


df_tenure["owner_share"] = df_tenure["owner_occupied"] / df_tenure["total_occupied"]


df_tenure.to_csv("atlanta_tenure_2019.csv", index=False)


# ============================================================
# 3. Household income distribution (B19001)
# ============================================================

income = []

for county in ATLANTA_COUNTIES:
    data = c.acs5.state_county_tract(
        fields=(
            "NAME",
            "B19001_001E",
            "B19001_002E",
            "B19001_007E",
            "B19001_009E",
            "B19001_013E",
            "B19001_014E",
            "B19001_017E",
        ),
        state_fips=GA_FIPS,
        county_fips=county,
        tract=Census.ALL,
    )

    income.extend(data)


df_income = pd.DataFrame(income)

df_income.rename(
    columns={
        "B19001_001E": "total_households",
        "B19001_002E": "income_under_10k",
        "B19001_007E": "income_30_35k",
        "B19001_009E": "income_40_45k",
        "B19001_013E": "income_75_100k",
        "B19001_014E": "income_100_125k",
        "B19001_017E": "income_200k_plus",
    },
    inplace=True,
)


df_income.to_csv("atlanta_income_2019.csv", index=False)


# ============================================================
# 4. Vacancy data (B25004)
# ============================================================

vacancy = []

for county in ATLANTA_COUNTIES:
    data = c.acs5.state_county_tract(
        fields=(
            "NAME",
            "B25004_001E",
            "B25004_002E",
            "B25004_004E",
        ),
        state_fips=GA_FIPS,
        county_fips=county,
        tract=Census.ALL,
    )

    vacancy.extend(data)


df_vacancy = pd.DataFrame(vacancy)

df_vacancy.rename(
    columns={
        "B25004_001E": "total_vacant",
        "B25004_002E": "vacant_for_rent",
        "B25004_004E": "vacant_for_sale",
    },
    inplace=True,
)


df_vacancy.to_csv("atlanta_vacancy_2019.csv", index=False)


# ============================================================
# 5. Merge into ABM master file
# ============================================================

stock = pd.read_csv("atlanta_housing_stock_2019.csv")
tenure = pd.read_csv("atlanta_tenure_2019.csv")
income = pd.read_csv("atlanta_income_2019.csv")
vacancy = pd.read_csv("atlanta_vacancy_2019.csv")


keys = ["state", "county", "tract"]


master = (
    stock.merge(
        tenure[keys + ["owner_occupied", "renter_occupied", "owner_share"]], on=keys
    )
    .merge(income.drop(columns="NAME"), on=keys)
    .merge(
        vacancy[keys + ["total_vacant", "vacant_for_rent", "vacant_for_sale"]], on=keys
    )
)


master["GEOID"] = master["state"] + master["county"] + master["tract"]


master.to_csv("atlanta_master_tract_2019.csv", index=False)


print(f"Master file: {master.shape[0]} tracts, " f"{master.shape[1]} variables")
