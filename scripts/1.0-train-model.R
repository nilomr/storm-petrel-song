# CONFIGURATION ------------------------------------------------------------

config <- config::get()
box::use(R / rplot[titheme])
progressr::handlers("cli")
options(scipen = 999)


# LOAD DATA ----------------------------------------------------------------

# Load the data without row names
data <- readr::read_csv(file.path(config$path$resources, "features.csv")) |>
    dplyr::select(-c(1))


# Function to balance classes
balance_classes <- function(data, subspecies = FALSE, min_class_size = 15) {
    data_equalclass <- data |>
        dplyr::group_by(ID) |>
        dplyr::sample_n(size = min_class_size, replace = TRUE) |>
        dplyr::ungroup()
    if (subspecies) {
        min_group_size <- min(table(data_equalclass$group))
        data_equalclass <- data_equalclass |>
            dplyr::group_by(group) |>
            dplyr::sample_n(size = min_group_size, replace = TRUE) |>
            dplyr::ungroup()
        y <- data_equalclass$group
    } else {
        y <- data_equalclass$ID
    }
    list(data_equalclass = data_equalclass, y = y)
}

# Function to train and fit Random Forest
train_random_forest <- function(X_train, y_train) {
    randforest <- randomForest::randomForest(
        x = X_train, y = y_train,
        ntree = 500, importance = TRUE,
        proximity = TRUE
    )
    randforest
}

# Function to collect metrics
collect_metrics <- function(randforest, X_test, y_test) {
    y_pred <- predict(randforest, X_test)
    conf_mat <- caret::confusionMatrix(y_pred, y_test)$table
    conf_mat <- prop.table(conf_mat)
    feature_importances <- randomForest::importance(randforest)
    report <- caret::confusionMatrix(y_pred, y_test)$byClass
    list(conf_mat = conf_mat, feature_importances = feature_importances, report = report)
}

# Drop any rows with missing values and print a report
missing_rows <- sum(apply(data, 1, function(x) any(is.na(x))))
print(paste("Dropped", missing_rows, "rows with missing values."))
data <- na.omit(data)

# Function to split and scale data
split_and_scale_data <- function(X, y) {
    trainIndex <- caret::createDataPartition(y, p = .75, list = FALSE)
    X_train <- X[trainIndex, ]
    X_test <- X[-trainIndex, ]
    y_train <- as.factor(y[trainIndex])
    y_test <- as.factor(y[-trainIndex])

    preProc <- caret::preProcess(X_train, method = c("center", "scale"))
    X_train <- predict(preProc, X_train)
    X_test <- predict(preProc, X_test)

    list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
}


# Train model
subspecies <- TRUE
conf_mat <- list()
feature_importances <- list()
report <- list()
samples <- list()
iterations <- 20
random_conf_mat <- list()
random_report <- list()

progressr::with_progress({
    p <- progressr::progressor(steps = iterations)
    for (i in 1:iterations) {
        # Balance classes (population)
        balanced_data <- balance_classes(data, subspecies, min_class_size = 20)
        data_equalclass <- balanced_data$data_equalclass
        y <- balanced_data$y

        samples[[i]] <- row.names(data_equalclass)
        X <- data_equalclass |> dplyr::select(-c(ID, group))
        strata <- data_equalclass$ID

        # Split and scale data
        split_data <- split_and_scale_data(X, y)

        # Train and fit Random Forest
        randforest <- train_random_forest(
            split_data$X_train,
            split_data$y_train
        )

        # Collect metrics
        metrics <- collect_metrics(
            randforest, split_data$X_test,
            split_data$y_test
        )
        conf_mat[[i]] <- metrics$conf_mat
        feature_importances[[i]] <- metrics$feature_importances
        report[[i]] <- metrics$report

        # Randomize y labels and retrain for baseline
        y_random <- sample(y)
        split_data_random <- split_and_scale_data(X, y_random)
        randforest_random <- train_random_forest(
            split_data_random$X_train,
            split_data_random$y_train
        )
        metrics_random <- collect_metrics(
            randforest_random, split_data_random$X_test,
            split_data_random$y_test
        )
        random_conf_mat[[i]] <- metrics_random$conf_mat
        random_report[[i]] <- metrics_random$report

        p()
    }
})

# Extract the Balanced Accuracy from the reports and plot

accs_df <- data.frame(
    balanced_accuracy = sapply(report, function(x) x[["Balanced Accuracy"]]),
    random_balanced_accuracy = sapply(
        random_report,
        function(x) x[["Balanced Accuracy"]]
    )
)
# turn into long format
accs_df <- tidyr::pivot_longer(
    accs_df,
    cols = c(balanced_accuracy, random_balanced_accuracy),
    names_to = "metric",
    values_to = "value"
)

# Create a ggplot object
ggplot2::ggplot(accs_df, ggplot2::aes(x = value, fill = metric)) +
    ggplot2::geom_histogram(ggplot2::aes(
        y = ggplot2::after_stat(density)
    ), bins = 20) +
    ggplot2::geom_density() +
    titheme()

# This works if subspieces is FALSE

# sumamrise the reports
report_df <- do.call(rbind, report) |>
    dplyr::as_tibble() |>
    dplyr::mutate(population = rep(rownames(report[[1]]), iterations)) |>
    dplyr::select(population, everything())
# remove 'Class: ' from the population column values
report_df$population <- gsub("Class: ", "", report_df$population)
# group the rows by the population column and calculate mean, se and 95% CI of the
# mean of each column
report_df <- report_df |>
    dplyr::group_by(population) |>
    dplyr::summarise_all(list(
        mean = ~ round(mean(., na.rm = TRUE), 2),
        se = ~ round(sd(., na.rm = TRUE) / sqrt(iterations), 2),
        CI_low = ~ round(mean(., na.rm = TRUE) - 1.96 * sd(., na.rm = TRUE) / sqrt(iterations), 2),
        CI_high = ~ round(mean(., na.rm = TRUE) + 1.96 * sd(., na.rm = TRUE) / sqrt(iterations), 2)
    )) |>
    dplyr::ungroup()

# clean the report by just having three columns per metric: metric_mean,
# metric_CI_low, metric_CI_high, which should then be printed as mean [CI_low, CI_high]
report_df <- report_df |> dplyr::select(population, contains("mean"), contains("CI"))

# sumamrise by creating a column per metric that contains a string wit the 'mean
# [CI_low, CI_high]'. Metrics are Sensitivity Specificity `Pos Pred Value` `Neg Pred Value` Precision Recall    F1        Prevalence `Detection Rate` `Detection Prevalence` `Balanced Accuracy`
report_df <- report_df |>
    dplyr::mutate(
        Precision = paste0(Precision_mean, " [", Precision_CI_low, ", ", Precision_CI_high, "]"),
        Recall = paste0(Recall_mean, " [", Recall_CI_low, ", ", Recall_CI_high, "]"),
        F1 = paste0(F1_mean, " [", F1_CI_low, ", ", F1_CI_high, "]"),
        `Balanced Accuracy` = paste0(`Balanced Accuracy_mean`, " [", `Balanced Accuracy_CI_low`, ", ", `Balanced Accuracy_CI_high`, "]")
    ) |>
    dplyr::select(population, Precision, Recall, F1, `Balanced Accuracy`)
print(report_df)
