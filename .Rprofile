RENV_CONFIG_SANDBOX_ENABLED <- FALSE

source("renv/activate.R")
#### -- Consistent File Downloads -- ####
if (.Platform$OS.type == "windows") {
  options(
    download.file.method = "wininet"
  )
} else {
  options(
    download.file.method = "libcurl"
  )
}

#### -- Set CRAN -- ####

options(
  CRAN = c("https://cran.rstudio.com/", stan = "https://mc-stan.org/r-packages/")
)

#### -- Factors Are Not Strings -- ####
options(
  stringsAsFactors = FALSE
)

#### -- Display -- ####
options(
  digits = 12, # number of significant digits to show by default
  width = 80 # console width
)

#### -- Time Zone -- ####
if (Sys.getenv("TZ") == "") Sys.setenv("TZ" = Sys.timezone())
if (interactive()) {
  message("Session Time: ", format(Sys.time(), tz = Sys.getenv("TZ"), usetz = TRUE))
}

#### -- Session -- ####
.First <- function() {
  if (interactive()) {
    cat("\n")
    utils::timestamp("", prefix = paste("##------ [", getwd(), "] ", sep = ""))
    cat("\nSuccessfully loaded .Rprofile at", base::date(), "\n")
  }
}

if (interactive()) {
  message("Session Info: ", utils::sessionInfo()[[4]])
  message("Session User: ", Sys.info()["user"])
}

options(
  prompt = "R > ",
  continue = "... "
)

#### -- Dev Tools -- ####
if (interactive()) {
  library(fs)
  library(devtools)
}

# Margeffects
options("marginaleffects_posterior_interval" = "hdi")

options(vsc.rstudioapi = TRUE) # added by `renvsc`
if (interactive() && Sys.getenv("TERM_PROGRAM") == "vscode") {
  if ("httpgd" %in% .packages(all.available = TRUE)) {
    options(vsc.plot = FALSE)
    options(device = function(...) {
      httpgd::hgd(silent = TRUE)
      .vsc.browser(httpgd::hgd_url(), viewer = "Beside")
    })
  }
} # added by `renvsc`

#### -- BOX -- ####
# find the repo root
repo_root <- fs::path_real(fs::path(rprojroot::find_rstudio_root_file()))
options(box.path = file.path(repo_root))
