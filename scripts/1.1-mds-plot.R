# ──── CONFIGURATION ──────────────────────────────────────────────────────────

config <- config::get()
box::use(R / rplot[titheme])
box::use(R / utils[print_time_elapsed])
box::use(patchwork)
progressr::handlers("cli")
message("Number of cores available: ", config$ncores)


# ──── FUNCTION DEFINITIONS ───────────────────────────────────────────────────


# ──── MAIN ───────────────────────────────────────────────────────────────────


# Load the data without row names
data <- readr::read_csv(file.path(config$path$resources, "features.csv")) |>
    dplyr::select(-c(1))

# Drop any rows with missing values and print a report
missing_rows <- sum(apply(data, 1, function(x) any(is.na(x))))
print(paste("Dropped", missing_rows, "rows with missing values."))
data <- na.omit(data)

# ──── PLOT MDS AND PCA ───────────────────────────────────────────────────────


data <- data |>
    dplyr::mutate(
        group = dplyr::case_when(
            ID %in% c(
                "faroes", "molene", "norway", "scotland", "iceland",
                "ireland", "wales", "montana_clara"
            ) ~ "pelagicus",
            ID %in% c("benidorm", "malta", "sardinia", "greece") ~ "melitensis"
        ),
        gengroup = as.factor(dplyr::case_when(
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

set.seed(42)
data_subsample <- data |>
    dplyr::group_by(gengroup) |>
    dplyr::sample_n(size = min(80, dplyr::n()), replace = FALSE) |>
    dplyr::ungroup()

# Prepare data
X <- data_subsample |> dplyr::select(-c(ID, group, gengroup))
y <- as.factor(data_subsample$group)
# scale the data
preProc <- caret::preProcess(X, method = c("center", "scale"))
X <- predict(preProc, X)

# Train and fit Random Forest
set.seed(42)
randforest <- randomForest::randomForest(
    x = X, y = y,
    ntree = 1500, proximity = TRUE, importance = TRUE
)

# Calculate MDS from the proximity matrix
mds <- cmdscale(as.dist(1 - randforest$proximity))
# add rownames to the MDS matrix (from the y labels)
rownames(mds) <- randforest$y

# ──── PLOT MDS by GROUP ──────────────────────────────────────────────────────

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

shapes <- list(
    "Macaronesia" = 24,
    "North-Eastern Atlantic" = 24,
    "Western Mediterranean" = 21,
    "Central Mediterranean" = 21,
    "Eastern Mediterranean" = 21
)

mds_df <- dplyr::as_tibble(mds) |>
    dplyr::mutate(pop = data_subsample$gengroup) |>
    dplyr::mutate(group = randforest$y) |>
    dplyr::mutate(pop = factor(pop, levels = names(pop_colors)))

mds_plot <-
    as.data.frame(mds_df) |>
    ggplot2::ggplot(ggplot2::aes(x = V1, y = V2, fill = pop, shape = group)) +
    ggplot2::geom_vline(xintercept = 0, color = "#464646") +
    ggplot2::geom_hline(yintercept = 0, color = "#464646") +
    ggplot2::geom_point(stroke = NA, size = 1.5, alpha = 0.9) +
    ggplot2::stat_ellipse(level = 0.8, geom = "polygon", alpha = 0.13) +
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
        fill = "Group"
    ) +
    titheme(aspect_ratio = 1) +
    ggplot2::theme(
        plot.subtitle = ggtext::element_markdown(),
        legend.key = ggplot2::element_blank(),
        # make axis titles  10pt
        axis.title = ggplot2::element_text(size = 10),
        panel.background = ggplot2::element_rect(fill = NA),
        panel.border = ggplot2::element_rect(color = "black", fill = NA),
        legend.spacing.y = ggplot2::unit(4, "cm")
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


# ──── PLOT SIMILARITY MATRIX ──────────────────────────────────────────────────


similarity_matrix <- as.matrix(randforest$proximity)
labels <- data_subsample$gengroup

rownames(similarity_matrix) <- labels
colnames(similarity_matrix) <- labels

# Convert the similarity matrix to a data frame with columns: ID1, ID2,
# similarity
similarity_df <- as.data.frame(as.table(similarity_matrix)) |> dplyr::as_tibble()
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

sim_mat_plot <- similarity_df |>
    dplyr::mutate(
        Var1 = factor(Var1, levels = names(pop_colors)),
        Var2 = factor(Var2, levels = names(pop_colors))
    ) |>
    ggplot2::ggplot(ggplot2::aes(x = Var1, y = Var2, fill = d)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_viridis_c(option = "turbo", na.value = "#3b3b3bb4") +
    ggplot2::scale_y_discrete(
        limits = names(pop_colors),
        labels = yaxis_labels
    ) +
    ggplot2::labs(
        x = "Genetic group",
        y = "Genetic group",
        title = "Mean RF acoustic distances",
        subtitle = "between genetic groups' purr calls",
        fill = "Mean\nsimilarity"
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

# ──── COMBINED PLOTS ─────────────────────────────────────────────────────────

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
