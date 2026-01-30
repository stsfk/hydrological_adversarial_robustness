# CAMELS-DE data processing
#
# Reads CAMELS-DE v1.0.0 catchment time series (P, T, Q) and simulated PET, merges them by date,
# performs a few basic quality checks (e.g., set negative Q to NA), filters to 1980-01-02–2020-12-31,
# removes catchments with insufficient non-missing Q in train/val/test periods, and writes split CSVs
# (train/val/test/all) plus selected catchment names to `./data/`.

if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  fs
)

# Read data ---------------------------------------------------------------

timeseries_files <- dir_ls("/Users/yang/Documents/projects/data/camels_de/Version 1.0.0/camels_de/timeseries/")

data <- vector("list", length(timeseries_files))

for (i in seq_along(timeseries_files)) {
  timeseries_file <- timeseries_files[[i]]
  catchment_name <- basename(timeseries_file) %>%
    str_split("_", simplify = TRUE) %>%
    {
      .[, ncol(.)]
    } %>% # take last column of the matrix
    str_sub(end = -5) # drop ".csv"

  timeseries <- read_csv(timeseries_file, show_col_types = FALSE) %>%
    select(date, Q = discharge_spec_obs, P = precipitation_mean, T = temperature_mean)

  timeseries_simulated_file <- paste0("/Users/yang/Documents/projects/data/camels_de/Version 1.0.0/camels_de/timeseries_simulated/CAMELS_DE_discharge_sim_", catchment_name, ".csv")
  timeseries_simulated <- read_csv(timeseries_simulated_file, show_col_types = FALSE) %>%
    select(date, PET = pet_hargreaves)

  data[[i]] <- tibble(
    catchment_name = catchment_name,
    data = list(timeseries %>% left_join(timeseries_simulated, by = join_by(date)) %>%
      select(date, P, T, PET, Q))
  )
}

data <- data %>% bind_rows()

save(data, file = "./data/camels_de_1.00.Rda")


# Split data --------------------------------------------------------------
load("./data/camels_de_1.00.Rda")

gc()

data_process <- data %>%
  unnest(data) %>%
  rename(Date = date)


# Quality checks ----------------------------------------------------------

# check forcing; no missing forcing
data_process %>%
  select(P:PET) %>%
  complete.cases() %>%
  sum()

# check forcing range; no apparent error
data_process %>%
  select(P:PET) %>%
  lapply(range)

# more Q data are available after 1980
data_process %>%
  group_by(Date) %>%
  summarise(Q_availablity = sum(!is.na(Q)) / n()) %>%
  ggplot(aes(Date, Q_availablity)) +
  geom_line()

# 4,038 Q is < 0, changed to NA
data_process <- data_process %>%
  mutate(Q = replace(Q, Q < 0, NA))


# Catchment selection -----------------------------------------------------

gc()

# use record from 1981-01-01 to 2020-12-31 for the modeling study
# the data from 1980-01-02 is for warm-up
data_process <- data_process %>%
  filter(
    Date >= ymd("1980-01-02"),
    Date <= ymd("2020-12-31")
  )

# training and validation are from 1981-01-01 to 2010-12-31, where data until 2000-12-31 are for training
# testing from 2011-01-01 to 2020-12-31

# all the forcing data is available, some of the Q data is missing
# catchments with missing Q records is stored in `incomplete_catchments`

minimal_required_Q_length <- 365 * 2 # at least 2 years of data should be available in each period

incomplete_catchment_train <- data_process %>%
  filter(
    Date <= ymd("2000-12-31"),
    Date >= ymd("1981-01-01")
  ) %>%
  group_by(catchment_name) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_name)

incomplete_catchment_val <- data_process %>%
  filter(
    Date >= ymd("2001-01-01"),
    Date <= ymd("2010-12-31")
  ) %>%
  group_by(catchment_name) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_name)

incomplete_catchment_test <- data_process %>%
  filter(
    Date >= ymd("2011-01-01"),
    Date <= ymd("2020-12-31")
  ) %>%
  group_by(catchment_name) %>%
  summarise(data = list(tibble(Q))) %>%
  mutate(
    n_complete_record = map_dbl(
      data, function(x) complete.cases(x) %>% sum()
    )
  ) %>%
  filter(n_complete_record < minimal_required_Q_length) %>%
  pull(catchment_name)

incomplete_catchments <-
  c(
    incomplete_catchment_train,
    incomplete_catchment_test,
    incomplete_catchment_val
  ) %>%
  unique()

# 1347 catchments left
data_process %>%
  filter(!(catchment_name %in% incomplete_catchments)) %>%
  pull(catchment_name) %>%
  unique() %>%
  length()

# keep only complete_catchments with sufficient Q records
data_process <- data_process %>%
  filter(!(catchment_name %in% incomplete_catchments))


# Split the data ----------------------------------------------------------

# training and validation are from 1981-01-01 to 2010-12-31
# forcing from 1980-01-02 is used for warm-up
data_train_val <- data_process %>%
  filter(
    Date <= ymd("2010-12-31"),
    Date >= ymd("1980-01-02")
  )

data_train_val %>%
  count(catchment_name) %>%
  pull(n) %>%
  unique() # length = 11322

# Q until 2000-12-31 is used for training
data_train <- data_process %>%
  filter(
    Date <= ymd("2000-12-31"),
    Date >= ymd("1980-01-02")
  )

data_train %>%
  count(catchment_name) %>%
  pull(n) %>%
  unique() # length = 7670

# Q from 2001-01-01 to 2010-12-31 is used for validation, forcing from 2000-01-02 is used for warm-up
data_val <- data_process %>%
  filter(
    Date >= ymd("2000-01-02"),
    Date <= ymd("2010-12-31")
  )

data_val %>%
  count(catchment_name) %>%
  pull(n) %>%
  unique() # length = 4017

# Q from 2011-01-01 is used for testing, forcing from 2010-01-01 is used for warm-up
data_test <- data_process %>%
  filter(
    Date >= ymd("2010-01-01"),
    Date <= ymd("2020-12-31")
  )

data_test %>%
  count(catchment_name) %>%
  pull(n) %>%
  unique() # length = 4018

# All the data used in modeling
data_all <- data_process %>%
  filter(
    Date >= ymd("1980-01-02"),
    Date <= ymd("2020-12-31")
  )

data_all %>%
  count(catchment_name) %>%
  pull(n) %>%
  unique() # length = 14975

# date range
data_train_val$Date %>% range() # from "1980-01-02" to "2010-12-31", with the first year for warm-up only
data_train$Date %>% range() # from "1980-01-02" to "2000-12-31", with the first year for warm-up only
data_val$Date %>% range() # from "2000-01-02" to "2010-12-31", with the first year for warm-up only
data_test$Date %>% range() # from "2010-01-01" to "2020-12-31", with the first year for warm-up only
data_all$Date %>% range() # from "1980-01-02" to "2020-12-31", with the first year for warm-up only


# Save data ---------------------------------------------------------------

data_train_val %>%
  arrange(catchment_name, Date) %>%
  write_csv(file = "./data/data_train_val_CAMELS_DE1.00.csv")

data_train %>%
  arrange(catchment_name, Date) %>%
  write_csv(file = "./data/data_train_CAMELS_DE1.00.csv")

data_val %>%
  arrange(catchment_name, Date) %>%
  write_csv(file = "./data/data_val_CAMELS_DE1.00.csv")

data_test %>%
  arrange(catchment_name, Date) %>%
  write_csv(file = "./data/data_test_CAMELS_DE1.00.csv")

data_all %>%
  arrange(catchment_name, Date) %>%
  write_csv(file = "./data/data_all_CAMELS_DE1.00.csv")

# write selected catchment names
data_all %>%
  count(catchment_name) %>%
  select(catchment_name) %>%
  write_csv(file = "./data/selected_catchments.csv")
