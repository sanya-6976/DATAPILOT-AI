# Real-world validation datasets

These small CSV fixtures support cross-domain validation of the FastAPI
dataset-upload pipeline. The fixtures are source data, not application
defaults; no production behavior is specialized for them.

| Domain | Fixture | Source and citation | License / usage |
|---|---|---|---|
| Education | `uci-student-performance.csv` | UCI Machine Learning Repository, [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), DOI [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). Cite: Cortez, P. (2008). *Student Performance* [Dataset]. | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on the UCI dataset page. Attribute the original dataset and authors. |
| Economics | `world-bank-india-population-2000-2024.csv` | World Bank Indicators API, India (`IND`), Population, total (`SP.POP.TOTL`): [API endpoint](https://api.worldbank.org/v2/country/IND/indicator/SP.POP.TOTL?format=json&per_page=100). | World Bank open data are generally available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the [World Bank public license terms](https://datacatalog.worldbank.org/public-licenses). Attribute the World Bank and check dataset-specific terms. |
| Weather | `noaa-laguardia-2024.csv` | NOAA/NCEI Global Surface Summary of the Day (GSOD), LaGuardia station `72503014732`, 2024: [source CSV](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2024/72503014732.csv), [metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516). | The metadata gives citation and distribution-liability terms, but does not state a specific license. Cite NOAA/NCEI and note its no-warranty terms; do not treat this note as a license grant. |

## Fixture details

- The UCI file is the repository's canonical comma-separated `data.csv`
  download. It contains 649 rows and 33 columns. The downloadable subject
  files also include a semicolon-separated variant; that separate variant is
  not used here.
- The World Bank API returns annual observations. The fixture selects records
  for 2000 through 2024 inclusive and stores only country, country code,
  indicator, indicator code, year, and value. It contains 25 rows and 6
  columns. The source API response reported `lastupdated` as `2026-10-08`.
- The NOAA fixture is the full 2024 daily station extract, not a resampled or
  hand-filtered subset. It contains 366 rows and 28 columns.

The source files were retrieved in November 2026. Dimensions above describe
the checked-in CSV fixtures. `backend/tests/test_real_world_datasets.py`
derives its dimensional and profile expectations from each fixture at test
time and does not encode dataset-specific quality findings.

## Interpretation notes

The integration test validates parsing, returned schema, profile consistency,
and quality-finding column references. It does not assert that every
domain-specific convention is understood. In this NOAA extract, `999.9`
appears in `STP`, `GUST`, and `SNDP`; the generic CSV pipeline treats these
as numeric values, not missing data, and they contribute to potential-outlier
findings. Interpreting such sentinels requires domain metadata and is not
performed by this validation. The `FRSHTT` field is also reported as
ambiguous between numeric and datetime interpretations; the upload preserves
its values rather than converting them.
