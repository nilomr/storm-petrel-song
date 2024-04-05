#' Print the time elapsed in minutes and seconds
#'
#' @param result The result object containing the time elapsed
#' @return None
#'
#' @examples
#' result <- c(1, 2, 123.45)
#' print_time_elapsed(result)
#'
#' @export
print_time_elapsed <- function(result) {
    time_elapsed <- result[3]
    minutes <- floor(time_elapsed / 60)
    seconds <- round(time_elapsed %% 60, 2)
    ifelse(minutes == 1, minute_text <- "minute", minute_text <- "minutes")
    message("Finished in ", minutes, " ", minute_text, " and ", seconds, " seconds")
}
