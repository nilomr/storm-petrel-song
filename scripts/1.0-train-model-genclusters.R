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
data <- data |> na.omit()

# Assign genetic population IDs
data = data |>
    dplyr::mutate(
        group = dplyr::case_when(
            ID %in% c(
                "faroes", "molene", "norway", "scotland", "iceland",
                "ireland", "wales", "montana_clara"
            ) ~ "pelagicus",
            ID %in% c("benidorm", "malta", "sardinia", "greece") ~ "melitensis"
        ),
        ID = as.factor(dplyr::case_when(
            ID == "montana_clara" ~ "Macaronesia",
            ID %in% c(
                "faroes", "molene", "norway", "scotland", "iceland",
                "ireland", "wales"
            ) ~ "North-Eastern Atlantic",
            ID == "benidorm" ~ "Western Mediterranean",
            ID %in% c("malta", "sardinia") ~ "Central Mediterranean",
            ID == "greece" ~ "Eastern Mediterranean"
        ))
    )

# Call the function
n <- 100
model_results <- train_model(data, TRUE, iterations = n, min_class_size = 20)


# ──── PLOT MDS AND PCA ───────────────────────────────────────────────────────

# train a random forest model on the full dataset (cap to 100 data points per ID)
set.seed(42)
data_subsample <- data |>
    dplyr::group_by(ID) |>
    # if an ID has mroe than 100 data points, sample 100
    dplyr::sample_n(size = min(80, dplyr::n()), replace = FALSE) |>
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
pelagicus <- c(
    "Macaronesia" = "#dd6438",
    "North-Eastern Atlantic" = "#56b87b",
    "Cantabrian Sea" = "#dba43a"
)

melitensis <- c(
    "Western Mediterranean" = "#4174bc",
    "Central Mediterranean" = "#9e53b8",
    "Eastern Mediterranean" = "#b8b8b8"
)

# from pop and using the pelagicus and melitensis palettes create a vector of
# colors
pop_colors <- c(pelagicus, melitensis)
names(pop_colors)

shapes <- list(
    "Macaronesia" = 24,
    "North-Eastern Atlantic" = 24,
    "Western Mediterranean" = 21,
    "Central Mediterranean" = 21,
    "Eastern Mediterranean" = 21
)

# Add the pop, group, and shapes to the MDS matrix data frame for plotting
mds_df <- dplyr::as_tibble(mds) |>
    dplyr::mutate(pop = data_subsample$ID) |>
    dplyr::mutate(group = randforest$y) |>
    dplyr::mutate(pop = factor(pop, levels = names(pop_colors)))



#──── PLOT MDS ───────────────────────────────────────────────────────────────

mds_plot <-
    as.data.frame(mds_df) |>
    ggplot2::ggplot(ggplot2::aes(x = V1, y = V2, fill = pop, shape = group)) +
    ggplot2::geom_point(stroke = NA, size = 1.5, alpha = 0.9) +
    ggplot2::stat_ellipse(level = 0.86, geom = "polygon", alpha = 0.15) +
    # plot the centroids of the populations
    ggplot2::geom_point(
        data = mds_df |>
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
    ggplot2::geom_vline(xintercept = 0, color = "#252525") +
    ggplot2::geom_hline(yintercept = 0, color = "#252525") +

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
    titheme(aspect_ratio = 1) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown(),
        legend.key = ggplot2::element_blank(),
        # make axis titles  10pt
        axis.title = ggplot2::element_text(size = 10),
        panel.background = ggplot2::element_rect(fill = NA),
        panel.border = ggplot2::element_rect(color = "black", fill = NA),
        legend.spacing.y = ggplot2::unit(4, 'cm')
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
    file.path(config$path$figures, "mds_plot_genclusters.png"),
    plot = mds_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)


#──── PLOT SIMILARITY MATRIX ──────────────────────────────────────────────────


similarity_matrix <- as.matrix(randforest$proximity)
labels <- data_subsample$ID

rownames(similarity_matrix) <- labels
colnames(similarity_matrix) <- labels

# Convert the similarity matrix to a data frame with columns: ID1, ID2,
# similarity
similarity_df <- as.data.frame(as.table(similarity_matrix))  |> dplyr::as_tibble()
similarity_df <- similarity_df |>
    dplyr::group_by(Var1, Var2) |>
    dplyr::summarise(d = mean(Freq)) |>
    dplyr::ungroup()

# Add rows for missing population
similarity_df <- similarity_df |>
    dplyr::bind_rows(
        dplyr::tibble(
            Var1 = "Cantabrian Sea",
            Var2 = unique(labels),
            d = NA
        ),
        dplyr::tibble(
            Var1 = unique(labels),
            Var2 = "Cantabrian Sea",
            d = NA
        ),
        dplyr::tibble(
            Var1 = "Cantabrian Sea",
            Var2 = "Cantabrian Sea",
            d = NA
        ),

    )

yaxis_labels <-
    gsub(" ", "\n", tools::toTitleCase(names(pop_colors)))

sim_mat_plot = similarity_df |>
    dplyr::mutate(
        Var1 = factor(Var1, levels = names(pop_colors)),
        Var2 = factor(Var2, levels = names(pop_colors))
    ) |>
    ggplot2::ggplot(ggplot2::aes(x = Var1, y = Var2, fill = d)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_viridis_c(option = "turbo", na.value = "#e0e0e07e") +
    ggplot2::scale_y_discrete(
        limits = names(pop_colors),
        labels = yaxis_labels
    ) +
    ggplot2::labs(
        x = "Genetic group",
        y = "Genetic group",
        title = "Mean RF acoustic distances",
        subtitle = "between genetic groups' purr calls",
        fill = "Mean *d*"
    ) +
    titheme(aspect_ratio = 1) +
    # rotate x axis labels 45 degrees
    ggplot2::theme(axis.text.x = ggplot2::element_text(
        angle = 45,
        hjust = 1,
        color = pop_colors
    )) +
    ggplot2::theme(axis.text.y = ggplot2::element_text(color = pop_colors)) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown(),
        legend.title = ggtext::element_markdown(),
                panel.background = ggplot2::element_rect(fill = NA)
    )

# Save the plot
pwidth <- 15
pheight <- pwidth
ggplot2::ggsave(
    file.path(config$path$figures, "similarity_matrix_genclusters.png"),
    plot = sim_mat_plot,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm"
)

#──── COMBINED PLOTS ─────────────────────────────────────────────────────────

# Join the plots vertically
joined_plots <- (mds_plot / sim_mat_plot)

# Save the plot
pwidth <- 15
pheight <- 20
ggplot2::ggsave(
    file.path(config$path$figures, "mds_sim_mat_genclusters.svg"),
    plot = joined_plots,
    width = pwidth,
    height = pheight,
    dpi = 300,
    units = "cm",
    device = svglite::svglite
)
