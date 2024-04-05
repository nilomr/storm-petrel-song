#' Custom ggplot2 theme
#'
#' This function returns a custom ggplot2 theme with a minimalistic design.
#'
#' @return A ggplot2 theme object.
#' @import ggplot2
#' @export
#' @examples
#' ggplot(mtcars, aes(x = wt, y = mpg)) +
#'     geom_point() +
#'     labs(title = "Custom ggplot2 theme", subtitle = "A minimalistic design") +
#'     titheme()
#'
#' @export
titheme <- function(aspect_ratio = NULL) {
    ggplot2::theme(
        # set aspect ratio
        aspect.ratio = aspect_ratio,
        text = ggplot2::element_text(
            size = 12, family = "Roboto Condensed",
            colour = "#313131"
        ),
        panel.grid.minor = ggplot2::element_blank(),
        axis.line = ggplot2::element_blank(),
        axis.ticks = ggplot2::element_blank(),
        panel.border = ggplot2::element_rect(
            fill = "transparent", colour = NA
        ),
        panel.background = ggplot2::element_rect(
            fill = "#f8f8f8", colour = NA
        ),
        panel.grid.major.y = ggplot2::element_line(
            color = "#ffffff",
            linewidth = 0.5,
            linetype = 1
        ),
        panel.grid.major.x = ggplot2::element_line(
            color = "#ffffff",
            linewidth = 0.5,
            linetype = 1
        ),
        axis.title.y = ggplot2::element_text(
            size = 12,
            margin = ggplot2::margin(t = 0, r = 10, b = 0, l = 0)
        ),
        axis.title.x = ggplot2::element_text(
            size = 12,
            margin = ggplot2::margin(t = 10, r = 0, b = 0, l = 0)
        ),
        strip.background = ggplot2::element_rect(
            fill = "transparent", colour = NA
        ),
        strip.text = ggplot2::element_text(
            size = 12, color = "#f3f3f3",
            margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 10)
        ),
        plot.subtitle = ggplot2::element_text(
            margin = ggplot2::margin(t = 0, r = 0, b = 5, l = 0)
        ),
        legend.background = ggplot2::element_rect(
            fill = "transparent", colour = NA
        ),
        legend.box.background = ggplot2::element_rect(
            fill = "transparent", colour = NA
        ),
        legend.key = ggplot2::element_rect(fill = "transparent", colour = NA),
        legend.spacing.y = ggplot2::unit(0.3, "lines"),
        # title to roboto condensed, size 12
        plot.title = ggplot2::element_text(
            size = 13, face = "bold", colour = "#313131",
            margin = ggplot2::margin(t = 0, r = 0, b = 5, l = 0)
        ),
        plot.background = ggplot2::element_rect(
            fill = "#ffffff", colour = NA
        ),
        axis.text = ggplot2::element_text(color = "#313131")
    )
}




#' Custom color palette
#'
#' This function returns a custom color palette with 2 or 3 colors.
#'
#' @param n An integer specifying the number of colors in the palette.
#' @return A named vector of color hex codes.
#' @examples
#' # get a 2-color palette
#' titpalette(2)
#'
#' # get a 3-color palette
#' titpalette(3)
#'
#' # error: n must be 2 or 3
#' titpalette(4)
#'
#' @export
titpalette <- function(n = 3, order = NULL) {
    # Define the lookup table
    lookup <- list(
        `2` = c("#459395", "#FDA638"),
        `3` = c("#459395", "#d36044", "#fdae38"),
        `4` = c("#459395", "#d35a4a", "#FDA638", "#674A40")
    )

    # Check if n is a valid key in the lookup table
    if (!as.character(n) %in% names(lookup)) {
        stop("n must be one of ", paste(names(lookup), collapse = ", "))
    }

    # Get the corresponding palette
    palette <- lookup[[as.character(n)]]

    # Reorder the palette if the order argument is specified
    if (!is.null(order)) {
        if (length(order) != length(palette)) {
            stop("order must have the same length as the palette")
        }
        palette <- palette[order]
    }

    # Return the palette
    return(palette)
}

#' @export
reds <- c("#d35a4a", "#db7b6e", "#e49c92", "#edbdb6")
#' @export
blues <- c("#539ca1", "#386e72")
#' @export
yellows <- c("#fdae38", "#c58439")
#' @export
persian <- c("#3b6b7e", "#577383")
