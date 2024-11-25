import time

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd

pandas2ri.activate()

robjects.r('''
suppressPackageStartupMessages(library(dplyr))

# Aggregate variable function
aggregate_variable <- function(samples, fixation_ids, variable, func) {
  sapply(fixation_ids, function(fid) {
    values <- samples[which(samples$fixation.id == fid), variable]
    func(values)
  })
}

# Detect and process events
detect.and.process.events <- function(samples, lambda = 6, pixels = FALSE, smooth.coordinates = TRUE) {
  required_columns <- c("x", "y", "trial", "time", "WinWidth", "WinHeight")
  if (!all(required_columns %in% colnames(samples))) {
    stop("Input data frame must have columns: 'x', 'y', 'trial', 'time', 'WinWidth', 'WinHeight'.")
  }

  if (smooth.coordinates) {
    x <- samples$x[c(1, nrow(samples))]
    y <- samples$y[c(1, nrow(samples))]
    kernel <- rep(1/3, 3)
    samples$x <- stats::filter(samples$x, kernel)
    samples$y <- stats::filter(samples$y, kernel)
    samples$x[c(1, nrow(samples))] <- x
    samples$y[c(1, nrow(samples))] <- y
  }

  samples <- detect.saccades(samples, lambda)
  fixations <- aggregate.fixations(samples)
  saccades <- aggregate.saccades(samples)

  return(list(fixations = fixations, saccades = saccades))
}

# Detect saccades using the Engbert-Kliegl algorithm
detect.saccades <- function(samples, lambda) {

  # Calculate horizontal and vertical velocities:
  vx <- stats::filter(samples$x, -1:1/2)
  vy <- stats::filter(samples$y, -1:1/2)

  # We don't want NAs, as they make our life difficult later
  # on.  Therefore, fill in missing values:
  vx[1] <- vx[2]
  vy[1] <- vy[2]
  vx[length(vx)] <- vx[length(vx)-1]
  vy[length(vy)] <- vy[length(vy)-1]

  msdx <- sqrt(stats::median(vx^2, na.rm=TRUE) - stats::median(vx, na.rm=TRUE)^2)
  msdy <- sqrt(stats::median(vy^2, na.rm=TRUE) - stats::median(vy, na.rm=TRUE)^2)

  radiusx <- msdx * lambda
  radiusy <- msdy * lambda

  sacc <- ((vx/radiusx)^2 + (vy/radiusy)^2) > 1

  samples$saccade <- ifelse(is.na(sacc), FALSE, sacc)
  samples$vx <- vx
  samples$vy <- vy

  samples

  }

# Aggregate fixations
aggregate.fixations <- function(samples) {
  saccade.events <- sign(c(0, diff(samples$saccade)))
  trial.numeric <- as.integer(factor(samples$trial))
  trial.events <- sign(c(0, diff(trial.numeric)))

  samples$fixation.id <- cumsum(saccade.events == -1 | trial.events == 1)
  fixation_ids <- unique(samples$fixation.id)

  events <- data.frame(
    trial = aggregate_variable(samples, fixation_ids, "trial", function(x) x[1]),
    x = aggregate_variable(samples, fixation_ids, "x", stats::median),
    y = aggregate_variable(samples, fixation_ids, "x", stats::median),
    duration = aggregate_variable(samples, fixation_ids, "time", function(x) max(x) - min(x)),
    dispersion = sqrt(aggregate_variable(samples, fixation_ids, "x", stats::mad)^2 +
                        aggregate_variable(samples, fixation_ids, "y", stats::mad)^2),
    stringsAsFactors = FALSE
  )
  return(events)
}

# Aggregate saccades
aggregate.saccades <- function(samples) {
  saccade_starts <- which(diff(c(FALSE, samples$saccade)) == 1)
  saccade_ends <- which(diff(c(samples$saccade, FALSE)) == -1)

  saccades <- data.frame()
  for (i in seq_along(saccade_starts)) {
    start_idx <- saccade_starts[i]
    end_idx <- saccade_ends[i]

    start_x <- samples$x[start_idx]
    start_y <- samples$y[start_idx]
    end_x <- samples$x[end_idx]
    end_y <- samples$y[end_idx]

    amplitude <- sqrt((end_x - start_x)^2 + (end_y - start_y)^2)
    angle <- atan2(end_y - start_y, end_x - start_x) * (180 / pi)

    saccades <- rbind(saccades, data.frame(
      trial = samples$trial[start_idx],
      duration = samples$time[end_idx] - samples$time[start_idx],
      amplitude = amplitude,
      angle = angle
    ))
  }
  return(saccades)
}

''')

detect_and_process_events = robjects.globalenv['detect.and.process.events']


def detect_events(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    start_time = time.time()
    # convert Pandas to R DataFrame
    r_df = robjects.pandas2ri.py2rpy(df)

    events_list = detect_and_process_events(r_df)
    fixations_r_df = events_list.rx2('fixations')
    saccades_r_df = events_list.rx2('saccades')

    # convert R DataFrame back to Pandas
    fixations_df = pandas2ri.rpy2py(fixations_r_df)
    saccades_df = pandas2ri.rpy2py(saccades_r_df)
    end_time = time.time()
    print(f"Done detecting events for window. Took {end_time - start_time} seconds.")

    return fixations_df, saccades_df
