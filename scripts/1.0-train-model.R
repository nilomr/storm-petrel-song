# ──── CONFIGURATION ──────────────────────────────────────────────────────────

config <- config::get()
box::use(R / rplot[titheme])
box::use(R / utils[print_time_elapsed])
box::use(patchwork)
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

#' Function to calculate mean, standard error, and confidence intervals
#'
#' This function takes a metric as input and calculates the mean, standard error,
#' and confidence intervals for that metric using the model results. Assumes
#' that the metric is normally distributed.
#'
#' @param metric The metric for which to calculate the statistics.
#' @param model_results The list of model results.
#' @return A vector containing the mean, lower, and upper CI.
calc_stats <- function(metric, model_results) {
    values <- extract_variable(model_results$report, metric)
    n <- length(values)
    mean_val <- mean(values)
    se_val <- sd(values) / sqrt(n)
    c(
        mean = mean_val, CI_low = mean_val - 1.96 * se_val,
        CI_high = mean_val + 1.96 * se_val
    )
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
n <- 500
model_results <- train_model(data, TRUE, iterations = n, min_class_size = 20)



#──── PLOT DISTRIBUTION OF (BALANCED) ACCURACY ───────────────────────────────


accs_df <- data.frame(
    acc = extract_variable(model_results$report, "Balanced Accuracy"),
    random_acc = extract_variable(
        model_results$random_report,
        "Balanced Accuracy"
    )
)
# Turn into long format
accs_df <- tidyr::pivot_longer(
    accs_df,
    cols = c(acc, random_acc),
    names_to = "metric",
    values_to = "value"
)

# Calculate the mean and CI of the balanced accuracy for each model
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
        ifelse(metric == "acc", "Balanced RF:\n", "Chance level:\n"),
        round(mean, 2),
        " [", round(CI_low, 2), ", ",
        round(CI_high, 2), "]"
    )) |>
    dplyr::pull(label)

# Create a ggplot object

# get the
set.seed(42)
accs_plot <-
    ggplot2::ggplot(accs_df, ggplot2::aes(x = value, fill = metric)) +
    ggplot2::geom_vline(
        xintercept = 0.5, linetype = "dashed",
        color = "#8f8f8f"
    ) +
    ggdist::stat_histinterval(
        ggplot2::aes(fill = metric), breaks = 10,
        n = 1000, slab_alpha = 0.7,
        interval_alpha = 0, point_alpha = 0
    ) +
    ggdist::stat_slab(
        ggplot2::aes(fill = metric),
        n = 1000, slab_alpha = 0.7
    ) +
    ggplot2::scale_fill_manual(
        values = c(
            "acc" = "#52483a",
            "random_acc" = "#b8a387"
        ),
        labels = acc_label
    ) +
    # add x ticks at 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
    ggplot2::scale_x_continuous(
        breaks = seq(0.3, 1, 0.1),
        limits = c(0.2, 1),
        expand = c(0, 0)
    ) +
    ggplot2::scale_y_continuous(limits = c(0, NA), expand = c(0, 0)) +
    ggplot2::labs(
        x = "Accuracy", y = "Density",
        fill = "Model",
        title = "RF Classifier Accuracy",
        subtitle = paste0(
            "Separating *H. p. pelagicus* and *melitensis* purr calls"
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


metrics_summary <- dplyr::as_tibble(t(sapply(
    metrics_names, calc_stats,
    model_results
))) |>
    dplyr::mutate(metric = metrics_names) |>
    dplyr::select(metric, dplyr::everything()) |>
    # round dbl columns to 2 decimal places
    dplyr::mutate_if(is.numeric, ~ round(., 3))

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
top5_features <- feature_importances |>
    dplyr::group_by(variable) |>
    dplyr::summarise(
        gini = mean(gini),
    ) |>
    dplyr::top_n(5, gini) |>
    dplyr::ungroup() |>
    dplyr::arrange(dplyr::desc(gini))

feat_imp_plot = feature_importances |>
    dplyr::filter(variable %in% top5_features$variable) |>
    ggplot2::ggplot(ggplot2::aes(x = gini, y = reorder(variable, gini))) +
    ggplot2::geom_jitter(
        fill = "#7a7a7a",
        stroke = NA,
        size = 1.5,
        alpha = 0.5,
        width = 0,
        height = 0.2,
        shape = 21,
    ) +
    ggdist::stat_pointinterval(
        ggplot2::aes(fill = gini),
        na.rm = TRUE,
        scale = 0.5,
        fill_type = "gradient",
        color = "black",
        interval_alpha = 0.5,
        point_alpha = 0.5
    ) +
    # add space above the top y category
    ggplot2::scale_y_discrete(
        labels = c( 
            "purr_ioi" = "Purr IOI",
            "mean_purr_duration" = "Purr Unit\nDuration",
            "purr_duration" = "Purr Duration",
            "song_length" = "Song Length",
            "breathe_duration" = "Breathe Note\nDuration"
        ),
        expand = ggplot2::expansion(0, 1)
    ) +
    # log the x axis
    ggplot2::scale_x_log10(
    ) +
    ggplot2::labs(
        y = "Feature",
        x = "Mean Decrease in\nGini Impurity (log10)",
        title = "RF Classifier Feature Importances",
        subtitle = "Top 5 features"
    ) +
    titheme(aspect_ratio = 1.7) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown()
    )


# ──── PLOT DISTRIBUTION OF TOP5 FEATURES ────────────────────────────────────

feat_palette <- c(
    "pelagicus" = "#7dc1ca",
    "melitensis" = "#e29b3d"
)

feat_dist_plot <-
    data |>
    dplyr::ungroup() |>
    dplyr::select(group, top5_features$variable) |>
    # subsample the data to have 30 data points per group
    dplyr::ungroup() |>
    # scale numeric columns
    dplyr::mutate(across(where(is.numeric), scale)) |>
    tidyr::pivot_longer(
        cols = -group,
        names_to = "variable",
        values_to = "value"
    ) |>
    # variable and group are factors
    dplyr::mutate(
        variable = as.factor(variable),
        group = as.factor(group)
    ) |>
    # arrange in the same order as top5_features$variable
    dplyr::mutate(
        variable = forcats::fct_relevel(
            variable, rev(top5_features$variable)
        )
    ) |>
    ggplot2::ggplot(ggplot2::aes(x = value, y = variable, color = group, fill = group)) +
    ggdist::stat_halfeye(n = 100, adjust = 1.5, slab_alpha = 0.8) +
    # limit x axis to -4, 4
    ggplot2::scale_x_continuous(limits = c(-4, 4)) +
    ggplot2::scale_y_discrete(labels = c(
        "Purr IOI" = "purr_ioi",
        "Purr Unit\nDuration" = "mean_purr_duration",
        "Purr Duration" = "purr_duration",
        "Call Length" = "song_length",
        "Breathe Note\nDuration" = "breathe_duration"

    )) +
    ggplot2::scale_fill_manual(
        values = feat_palette,
        labels = c("melitensis" = "H. p. melitensis", "pelagicus" = "H. p. pelagicus")
    ) +
    ggplot2::scale_color_manual(
        values = colorspace::darken(feat_palette, 0.2),
        guide = "none"
    ) +
    ggplot2::labs(
        x = "Normalised\nFeature Value",
        y = "Density",
        title = "Purr Call Features",
        subtitle = "Top 5 features, scaled and centered",
        fill = ""
    ) +
    titheme(aspect_ratio = 1.6) +
    # remove y axis title and labels
    ggplot2::theme(
        axis.title.y = ggplot2::element_blank(),
        axis.text.y = ggplot2::element_blank(),
        axis.ticks.y = ggplot2::element_blank(),
        # make legend text cursive
        legend.text = ggplot2::element_text(face = "italic")
    )


# join the plots
feat_imp_plot + feat_dist_plot



# ──── PLOT MDS AND PCA ───────────────────────────────────────────────────────

# train a random forest model on the full dataset (cap to 100 data points per ID)
set.seed(42)
data_subsample <- data |>
    dplyr::group_by(ID) |>
    # if an ID has mroe than 100 data points, sample 100
    dplyr::sample_n(size = min(60, dplyr::n()), replace = FALSE) |>
    dplyr::ungroup()

# Prepare data
X = data_subsample |> dplyr::select(-c(ID, group))
y = as.factor(data_subsample$group)
# scale the data
preProc <- caret::preProcess(X, method = c("center", "scale"))
X <- predict(preProc, X)

# Train and fit Random Forest
set.seed(42)
randforest <- randomForest::randomForest(
    x =X , y = y,
    ntree = 1000, importance = TRUE,
    proximity = TRUE,
)

# Calculate MDS from the proximity matrix
mds <- cmdscale(as.dist(1 - randforest$proximity))
# add rownames to the MDS matrix (from the y labels)
rownames(mds) <- randforest$y

# set color palette based on population
# create a cold palette for pelagicus and a warm palette for melitensis
pelagicus = c(
    "norway" = "#9b5ac4", "iceland" = "#8e68bc", "faroes" = "#8177b3",
    "scotland" = "#7485ab", "ireland" = "#6894a3", "wales" = "#5ba29b",
    "molene" = "#4eb192", "montana_clara" = "#41bf8a", "mouro" = "#41bf52"
)

melitensis = c(
    "benidorm" = "#e9b423", "sardinia" = "#da700d",
    "malta" = "#ce6a5b", "greece" = "#b83647"
)

# from pop and using the pelagicus and melitensis palettes create a vector of
# colors
pop_colors = c(pelagicus, melitensis)
names(pop_colors)

# create a vector of 21 and 22 for the shapes that correspond to y
# (pelagicus/melitensis), repeating as necessary
shapes = list(
    "faroes" = 24, "molene" = 24, "norway" = 24, "scotland" = 24, "mouro" = 24,
    "iceland" = 24, "ireland" = 24, "montana_clara" = 24, "wales" = 24,
    "greece" = 21, "benidorm" = 21, "sardinia" = 21, "malta" = 21
)

# Add the pop, group, and shapes to the MDS matrix data frame for plotting
mds <- as.data.frame(mds) |>
    dplyr::mutate(pop = data_subsample$ID) |>
    dplyr::mutate(group = randforest$y) |>
    dplyr::mutate(pop = factor(pop, levels = names(pop_colors)))

# Plot using ggplot2
mds_plot <-
    as.data.frame(mds) |>
    ggplot2::ggplot(ggplot2::aes(x = V1, y = V2, fill = pop, shape = group)) +
    ggplot2::geom_point(stroke = NA, size = 1.5, alpha = 0.9) +
    ggplot2::stat_ellipse(level = 0.86, geom = "polygon", alpha = 0.09) +
    # plot the centroids of the populations
    ggplot2::geom_point(
        data = mds |>
            dplyr::group_by(pop) |>
            dplyr::mutate(
                V1 = mean(V1),
                V2 = mean(V2)
            ) |> # remove repeated rows
            dplyr::distinct(),
        ggplot2::aes(x = V1, y = V2, fill = pop, shape = group),
        color = "#464646",
        size = 3
    ) +
    ggplot2::scale_fill_manual(
        values = pop_colors,
        breaks = names(pop_colors),
        labels = function(x) gsub("_", " ", tools::toTitleCase(x))
    ) +
    ggplot2::scale_shape_manual(
        values =
            c("pelagicus" = 24, "melitensis" = 21), guide = "none"
    ) +
    ggplot2::labs(
        x = "MDS1",
        y = "MDS2",
        title = "MDS RF Proximity Matrix",
        subtitle = "*H. p. pelagicus* and *melitensis* purr calls",
        fill = "Population"
    ) +
    # inverse x axis
    ggplot2::scale_x_reverse() +
    ggplot2::scale_y_reverse() +
    titheme(aspect_ratio = 1) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown(),
        legend.key = ggplot2::element_blank(),
        # make axis titles  10pt
        axis.title = ggplot2::element_text(size = 10)
    ) +
    ggplot2::guides(
        fill = ggplot2::guide_legend(
            override.aes = list(shape = shapes, size = 2, stroke = NA)
        )
    )

# Save the plot
pwidth <- 15
pheight <- pwidth
ggplot2::ggsave(
    file.path(config$path$figures, "mds_plot.png"),
    plot = mds_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)

# Plot a PCA from the original data (w/ a subset of features)
pca_sub = prcomp(X[, top5_features$variable], scale = TRUE)
pca_df = as.data.frame(pca_sub$x)
pca_df$group = y

# Extract PC loadings for plotting
loadings <- as.data.frame(pca_sub$rotation[, 1:2]) |>
    tibble::rownames_to_column("variable")

# Plot using ggplot2
pca_plot <-
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
    ggplot2::scale_fill_manual(values = feat_palette) +
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

# Save the plot
pwidth <- 15
pheight <- pwidth
ggplot2::ggsave(
    file.path(config$path$figures, "pca_plot.png"),
    plot = pca_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)


# ──── COMBINE PLOTS ──────────────────────────────────────────────────────────

full_plot = (accs_plot + mds_plot + feat_dist_plot)

# Save the plot
pwidth <- 43
pheight <- 15

ggplot2::ggsave(
    file.path(config$path$figures, "full_plot.png"),
    plot = full_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)
caption1 = "\n\nYou'll need to push plot contents down a bit to centre them on the panel."
caption2 = "I've left titles and subtitles in place for reference, but I'll guess you'll want to remove them."
caption = paste(caption1, caption2, sep = "\n")
feature_plot = feat_imp_plot + feat_dist_plot +
    ggplot2::annotate("text", x = 0, y = 0, label = " ", hjust = 0, vjust = -1) + # dirty hack to align y-axis label positions
    patchwork::plot_annotation(
        caption = caption,
        theme = ggplot2::theme(plot.caption = ggplot2::element_text(hjust = 0.5))
    )


ggplot2::ggsave(
    file.path(config$path$figures, "feature_plot.png"),
    plot = feature_plot,
    width = pwidth/2,
    height = pheight,
    dpi = 300,
    units = "cm",
)

# save to svg
ggplot2::ggsave(
    file.path(config$path$figures, "feature_plot.svg"),
    plot = feature_plot,
    width = pwidth/2,
    height = pheight,
    units = "cm",
)
