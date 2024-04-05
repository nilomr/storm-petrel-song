# ──── CONFIGURATION ──────────────────────────────────────────────────────────

config <- config::get()
box::use(R / rplot[titheme])
box::use(R / utils[print_time_elapsed])
progressr::handlers("cli")
message("Number of cores available: ", config$ncores)


# ──── FUNCTION DEFINITIONS ───────────────────────────────────────────────────

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
    list(
        conf_mat = conf_mat, feature_importances = feature_importances,
        report = report
    )
}



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
# Function to balance classes and split data
prepare_data <- function(data, subspecies, min_class_size) {
    # Balance classes (population)
    balanced_data <- balance_classes(data, subspecies, min_class_size)
    data_equalclass <- balanced_data$data_equalclass
    y <- balanced_data$y
    X <- data_equalclass |> dplyr::select(-c(ID, group))
    # Split and scale data
    split_data <- split_and_scale_data(X, y)
    list(split_data = split_data, samples = row.names(data_equalclass))
}

# Function to train model and collect metrics
train_and_evaluate <- function(split_data) {
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
    metrics
}
# Function to train model
train_model <- function(data, subspecies, iterations, min_class_size) {
    progressr::with_progress({
        p <- progressr::progressor(steps = iterations)
        time <- system.time({
            itdata_results <- furrr::future_map(seq_len(iterations), function(i) {
                itdata <- list()
                # Prepare data
                prepared_data <- prepare_data(data, subspecies, min_class_size)
                itdata[["samples"]] <- prepared_data$samples

                # Train and evaluate model
                metrics <- train_and_evaluate(prepared_data$split_data)
                itdata[["conf_mat"]] <- metrics$conf_mat
                itdata[["feature_importances"]] <- metrics$feature_importances
                itdata[["report"]] <- metrics$report

                # Randomize y labels and retrain for baseline
                y_random <- sample(prepared_data$split_data$y_train)
                split_data_random <- split_and_scale_data(
                    prepared_data$split_data$X_train, y_random
                )
                metrics_random <- train_and_evaluate(split_data_random)
                itdata[["random_conf_mat"]] <- metrics_random$conf_mat
                itdata[["random_report"]] <- metrics_random$report

                p()
                return(itdata)
            }, .options = furrr::furrr_options(seed = TRUE))
        })
        # Pretty print time elapsed
        print_time_elapsed(time)
    })
    # now clean the results so that they can be easily summarized (eg the
    # returned object has object$conf_mat, object$feature_importances, etc):
    report <- lapply(itdata_results, function(x) x[["report"]])
    random_report <- lapply(itdata_results, function(x) x[["random_report"]])
    conf_mat <- lapply(itdata_results, function(x) x[["conf_mat"]])
    random_conf_mat <- lapply(itdata_results, function(x) x[["random_conf_mat"]])
    feature_importances <- lapply(itdata_results, function(x) x[["feature_importances"]])
    return(list(
        report = report,
        random_report = random_report,
        conf_mat = conf_mat,
        random_conf_mat = random_conf_mat,
        feature_importances = feature_importances
    ))
}

#' Extracts a specific variable from a list of model results.
#'
#' This function takes a list of model results and extracts a specific variable
#' from each result. The variable to be extracted is specified by the 'variable'
#' parameter.
#'
#' @param model_results A list of model results, where each result is a list
#' containing the model output. @param variable The name of the variable to be
#' extracted.
#'
#' @return A vector containing the extracted values of the specified variable
#' from each model result.
#'
#' @examples model_results <- list(result1 = list(a = 1, b = 2), result2 =
#' list(a = 3, b = 4)) extract_variable(model_results, "a")
#'
#' @export
extract_variable <- function(model_results, variable) {
    sapply(model_results, function(x) x[[variable]])
}


#──── MAIN ───────────────────────────────────────────────────────────────────


# Load the data without row names
data <- readr::read_csv(file.path(config$path$resources, "features.csv")) |>
    dplyr::select(-c(1))

# Drop any rows with missing values and print a report
missing_rows <- sum(apply(data, 1, function(x) any(is.na(x))))
print(paste("Dropped", missing_rows, "rows with missing values."))
data <- na.omit(data)


# Call the function
n <- 100
model_results <- train_model(data, TRUE, iterations = n, min_class_size = 20)

# Plot distribution of (balanced) accuracy

accs_df <- data.frame(
    acc = extract_variable(model_results$report, "Balanced Accuracy"),
    random_acc = extract_variable(model_results$random_report, "Balanced Accuracy")
)
# turn into long format
accs_df <- tidyr::pivot_longer(
    accs_df,
    cols = c(acc, random_acc),
    names_to = "metric",
    values_to = "value"
)

# calculate the mean and CI of the balanced accuracy for each model
acc_summary <- accs_df |>
    dplyr::group_by(metric) |>
    dplyr::summarise(
        mean = mean(value),
        se = sd(value) / sqrt(dplyr::n()),
        CI_low = mean - 1.96 * se,
        CI_high = mean + 1.96 * se
    ) |>
    dplyr::ungroup()

acc_label <- acc_summary |>
    dplyr::mutate(label = paste0(
        ifelse(metric == "acc", "Balanced RF:\n", "Randomised RF:\n"),
        round(mean, 2),
        " [", round(CI_low, 2), ", ",
        round(CI_high, 2), "]"
    )) |>
    dplyr::pull(label)

# Create a ggplot object
accs_plot <-
    ggplot2::ggplot(accs_df, ggplot2::aes(x = value, fill = metric)) +
    ggplot2::geom_vline(
        xintercept = 0.5, linetype = "dashed",
        color = "#8f8f8f"
    ) +
    ggplot2::geom_histogram(ggplot2::aes(
        y = ggplot2::after_stat(density) / n
    ), bins = 30, alpha = 0.3, color = NA, position = "identity") +
    ggplot2::geom_density(ggplot2::aes(
        y = ggplot2::after_stat(density) / n
    ), alpha = 0.7, color = NA, adjust = 1.5) +
    # Then, modify the labels in the scale_fill_manual function
    ggplot2::scale_fill_manual(
        values = c(
            "acc" = "#c08122",
            "random_acc" = "#4690a7"
        ),
        labels = acc_label
    ) +
    # add x ticks at 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
    ggplot2::scale_x_continuous(
        breaks = seq(0.3, 1, 0.1),
        limits = c(0.3, 1),
        expand = c(0, 0)
    ) +
    ggplot2::scale_y_continuous(limits = c(0, 0.6), expand = c(0, 0)) +
    ggplot2::labs(
        x = "Accuracy", y = "Density",
        fill = "Model",
        title = "Classifier Accuracy Distribution",
        subtitle = paste0(
            "*H. p. pelagicus* vs *melitensis* purr calls"
        )
    ) +
    titheme(aspect_ratio = 0.75) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown()
    ) +
    ggplot2::guides(fill = ggplot2::guide_legend(reverse = TRUE, byrow = TRUE))

# Save the plot
pwidth <- 15
# calculate height based on aspect ratio
pheight <- pwidth * 0.75
ggplot2::ggsave(
    file.path(config$path$figures, "accuracy_distribution.png"),
    plot = accs_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)

#──── EXTRACT BINARY METRICS ─────────────────────────────────────────────────


# Extract the names of the metrics
metrics_names <- names(model_results$report[[1]])

# Function to calculate mean, standard error, and confidence intervals
calc_stats <- function(metric) {
    values <- extract_variable(model_results$report, metric)
    n <- length(values)
    mean_val <- mean(values)
    se_val <- sd(values) / sqrt(n)
    c(mean = mean_val, CI_low = mean_val - 1.96 * se_val,
      CI_high = mean_val + 1.96 * se_val)
}

metrics_summary <- dplyr::as_tibble(t(sapply(metrics_names, calc_stats))) |>
    dplyr::mutate(metric = metrics_names) |>
    dplyr::select(metric, dplyr::everything()) |>
    # round dbl columns to 2 decimal places
    dplyr::mutate_if(is.numeric, ~round(., 2))

# Save to a CSV file
readr::write_csv(metrics_summary, file.path(
    config$path$reports,
    "binary_metrics_summary.csv"
))


# ──── PLOT THE FEATURE IMPORTANCES ───────────────────────────────────────────

feature_importances <- do.call(rbind, lapply(
    model_results$feature_importances,
    function(x) {
        as.data.frame(x) |>
            tibble::rownames_to_column("variable") |>
            dplyr::as_tibble() |>
            dplyr::select(variable, gini = MeanDecreaseGini)
    }
))

# plot the 10 most important features (on average)
top10_features <- feature_importances |>
    dplyr::group_by(variable) |>
    dplyr::summarise(
        gini = mean(gini),
    ) |>
    dplyr::top_n(10, gini) |>
    dplyr::ungroup() |>
    dplyr::arrange(dplyr::desc(gini))

feat_imp_plot = feature_importances |>
    dplyr::filter(variable %in% top10_features$variable) |>
    ggplot2::ggplot(ggplot2::aes(x = reorder(variable, gini), y = gini)) +
    ggdist::stat_gradientinterval(
        ggplot2::aes(fill = gini),
        na.rm = TRUE,
        scale = 0.5,
        fill_type = "gradient",
        color
    ) +
    ggplot2::geom_jitter(
        fill = "#3e8581",
        stroke = NA,
        size = 1,
        alpha = 0.5,
        width = 0.2,
        shape = 21,
    ) +
    ggplot2::coord_flip() +
    ggplot2::labs(
        x = "Feature",
        y = "Mean Decrease in Gini Impurity",
        title = "RF Classifier Feature Importances",
        subtitle = "Top 10 features by mean decrease in Gini Impurity"
    ) +
    titheme(aspect_ratio = 0.75) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown()
    )

# Save the plot
pwidth <- 15
pheight <- pwidth * 0.75
ggplot2::ggsave(
    file.path(config$path$figures, "feature_importances.png"),
    plot = feat_imp_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)

# Plot MDS and PCA

# train a random forest model on the full dataset.
data_subsample <- data |>
    # sample a maximum of 30 rows per ID
    dplyr::group_by(ID) |>
    dplyr::sample_n(size = 30, replace = TRUE) |>
    dplyr::ungroup()

# Prepare data
X = data_subsample |> dplyr::select(-c(ID, group))
y = as.factor(data_subsample$group)
pop = data_subsample$ID
# scale the data
preProc <- caret::preProcess(X, method = c("center", "scale"))
X <- predict(preProc, X)

# Train and fit Random Forest
randforest <- train_random_forest(X, y)

# Calculate MDS from the proximity matrix
mds <- cmdscale(as.dist(1 - randforest$proximity))
# add rownames to the MDS matrix (from the y labels)
rownames(mds) <- randforest$y


# set color palette based on population
# create a cold palette for pelagicus and a warm palette for melitensis
pelagicus = c(
    "faroes" = "#d7fff7", "molene" = "#05a3a6", "norway" = "#006373",
    "scotland" = "#bfd5c9", "iceland" = "#5e5ca5", "ireland" = "#60b1a3",
    "montana_clara" = "#617caf", "wales" = "#9273af"
)

melitensis = c(
    "greece" = "#9b5847", "benidorm" = "#b35a20",
    "sardinia" = "#f56727", "malta" = "#e8891d"
)

# from pop and using the pelagicus and melitensis palettes create a vector of
# colors
pop_colors = c(pelagicus, melitensis)
names(pop_colors)

# create a vector of 21 and 22 for the shapes that correspond to y
# (pelagicus/melitensis), repeating as necessary
shapes = list(
    "faroes" = 21, "molene" = 21, "norway" = 21, "scotland" = 21,
    "iceland" = 21, "ireland" = 21, "montana_clara" = 21, "wales" = 21,
    "greece" = 22, "benidorm" = 22, "sardinia" = 22, "malta" = 22
)

# Plot using ggplot2
mds_plot <-
as.data.frame(mds) |>
    dplyr::mutate(group = randforest$y) |>
    ggplot2::ggplot(ggplot2::aes(x = V1, y = V2, fill = pop, shape = y)) +
    ggplot2::geom_point(stroke = NA, size = 2) +
    ggplot2::stat_ellipse(level = 0.86, geom = "polygon", alpha = 0.2) +
    ggplot2::scale_fill_manual(
        values = pop_colors,
        breaks = names(pop_colors),
        labels = function(x) gsub("_", " ", tools::toTitleCase(x))
    ) +
    ggplot2::scale_shape_manual(values = c(21, 22), guide = FALSE) + # 21 is circle, 22 is square
    ggplot2::labs(
        x = "MDS1",
        y = "MDS2",
        title = "Multidimensional Scaling of Random Forest Proximity Matrix",
        subtitle = "*H. p. pelagicus* and *melitensis* purr calls",
        fill = "Population"
    ) +
    titheme(aspect_ratio = 1) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown(),
        legend.key = ggplot2::element_blank() # remove legend background
    ) +
    ggplot2::guides(
        fill = ggplot2::guide_legend(
            override.aes = list(shape = shapes)
        )
    ) # match legend point shape to plot


# Save the plot
pwidth <- 15
pheight <- pwidth * 0.75
ggplot2::ggsave(
    file.path(config$path$figures, "mds_plot.png"),
    plot = mds_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)


# plot the frequency distribution of the top10_features in the original data

data |>
    dplyr::select(group, top10_features$variable) |>
    # scale numeric columns
    dplyr::mutate(across(where(is.numeric), scale)) |>
    tidyr::pivot_longer(
        cols = -group,
        names_to = "variable",
        values_to = "value"
    ) |>
    ggplot2::ggplot(
        ggplot2::aes(x = value, y= as.factor(variable), fill = group)) +
    ggplot2::geom_point(
        ggplot2::aes(color = group),
        stroke = NA,
        size = 2,
        alpha = 0.5,
        shape = 21
    ) +
    ggplot2::labs(
        x = "Feature Value",
        y = "Density",
        title = "Feature Value Distribution",
        subtitle = "Top 10 features by mean decrease in Gini Impurity"
    )


# Plot a PCA from the original data (w/ a subset of features)

pca_sub = prcomp(X[, top10_features$variable], scale = TRUE)
pca_df = as.data.frame(pca_sub$x)
pca_df$group = y


# Extract PC loadings for plotting
loadings <- as.data.frame(pca_sub$rotation[, 1:2]) |>
    tibble::rownames_to_column("variable")


# Plot using ggplot2
pca_df |>
    dplyr::as_tibble() |>
    ggplot2::ggplot(ggplot2::aes(x = PC1, y = PC2)) +
    ggplot2::geom_segment(
        data = loadings,
        ggplot2::aes(x = 0, y = 0, xend = PC1 * 3, yend = PC2 * 3),
        arrow = ggplot2::arrow(length = ggplot2::unit(0.3, "cm")),
        color = "#313131"
    ) +
    ggplot2::annotate(
        "text", x = loadings$PC1 * 3, y = loadings$PC2 * 3,
        label = loadings$variable, size = 3
    ) +
    ggplot2::geom_point(ggplot2::aes(x = PC1, y = PC2, fill = group),
    stroke = NA, size = 2, shape=21) +
    ggplot2::stat_ellipse(ggplot2::aes(x = PC1, y = PC2, fill = group),
    level = 0.86, geom = "polygon", alpha = 0.2) +
    # ggplot2::scale_fill_manual(
    #     values = c("#c08122", "#4690a7"),
    #     labels = c("H. p. pelagicus", "H. p. melitensis")
    # ) +
    ggplot2::labs(
        x = "PC1",
        y = "PC2",
        title = "Principal Component Analysis of Scaled Features",
        subtitle = "*H. p. pelagicus* and *melitensis* purr calls",
        fill = "Group"
    ) +
    titheme(aspect_ratio = 0.75) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown(),
        legend.key = ggplot2::element_blank() # remove legend background
    )
