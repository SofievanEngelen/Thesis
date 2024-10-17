import time

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd

pandas2ri.activate()

robjects.r('''
suppressPackageStartupMessages(library(dplyr))

# FIXATION + BLINK DETECTION
detect.and.process.events <- function(samples, lambda = 6, smooth.coordinates = TRUE, smooth.saccades = FALSE) {
  
  # Validate input columns
  if (!all(c("x", "y", "trial", "time") %in% colnames(samples))) {
    stop("Input data frame needs columns 'x', 'y', 'trial', and 'time'.")
  }
  
  # Check chronological order
  if (!all(with(samples, tapply(time, trial, function(x) all(diff(x) > 0))))) {
    stop("Samples need to be in chronological order within trial.")
  }
  
  # Keep only necessary columns
  samples <- samples %>% select(x, y, trial, time, Participant, Paragraph)
  
  # Smooth coordinates if specified
  if (smooth.coordinates) {
    x <- samples$x[c(1, nrow(samples))]
    y <- samples$y[c(1, nrow(samples))]
    kernel <- rep(1/3, 3)
    samples$x <- stats::filter(samples$x, kernel)
    samples$y <- stats::filter(samples$y, kernel)
    samples$x[c(1, nrow(samples))] <- x
    samples$y[c(1, nrow(samples))] <- y
  }
  
  # Detect saccades
  samples <- detect.saccades(samples, lambda, smooth.saccades)
  
  # Check if any saccades detected
  if (all(!samples$saccade)) {
    stop("No saccades were detected. Something went wrong.")
  }
  
  # Aggregate events
  # all_events <- aggregate.events(samples)
  # all_events$event <- label.blinks.artifacts(all_events)
  all_events <- aggregate_and_label_events(samples)

  # Process Fixations and Blinks using dplyr for efficiency
  fixations_df <- all_events %>%
    dplyr::filter(event == "fixation") %>%
    reframe(
      trial = trial,
      paragraph = paragraph,
      participant = participant,
      x = x,
      y = y,
      duration = end - start, # Duration
      mad = sqrt(mad.x^2 + mad.y^2),  # Dispersion
    )
  
  blinks_df <- all_events %>%
    dplyr::filter(event == "blink") %>%
    reframe(
      trial = trial,
      paragraph = paragraph,
      participant = participant,
      duration = end - start,
    )
  
  # Return the processed data as a list
  return(list(fixations = fixations_df, blinks = blinks_df))
}

aggregate_and_label_events <- function(samples) {

  # Identify saccade events: 1 marks the start of a saccade, -1 marks the start of a fixation
  saccade.events <- sign(c(0, diff(samples$saccade)))
  trial.numeric  <- as.integer(factor(samples$trial))
  trial.events   <- sign(c(0, diff(trial.numeric)))

  # New fixations start when a saccade ends or a trial ends
  samples$fixation.id <- cumsum(saccade.events == -1 | trial.events == 1)
  samples$t2 <- samples$time
  samples$t2 <- ifelse(trial.events == 1, NA, samples$t2)
  samples$t2 <- samples$t2[2:(nrow(samples) + 1)]
  samples$t2 <- with(samples, ifelse(is.na(t2), time, t2))

  # Aggregate events
  events <- with(samples, data.frame(
    trial   = tapply(trial, fixation.id, function(x) x[1]),
    paragraph = first(Paragraph),
    participant = first(Participant),
    x       = tapply(x, fixation.id, stats::median),
    y       = tapply(y, fixation.id, stats::median),
    start   = tapply(time, fixation.id, min),
    end     = tapply(t2, fixation.id, function(x) max(x, na.rm = TRUE)),
    mad.x   = tapply(x, fixation.id, stats::mad),
    mad.y   = tapply(y, fixation.id, stats::mad),
    stringsAsFactors = FALSE
  ))

  # Calculate event duration
  events$dur <- events$end - events$start

  # Blink and artifact detection
  lsdx <- log10(events$mad.x)
  lsdy <- log10(events$mad.y)
  median.lsdx <- stats::median(lsdx, na.rm = TRUE)
  median.lsdy <- stats::median(lsdy, na.rm = TRUE)
  mad.lsdx <- stats::mad(lsdx, na.rm = TRUE)
  mad.lsdy <- stats::mad(lsdy, na.rm = TRUE)

  # Dispersion thresholds for blink detection
  threshold.lsdx_low <- median.lsdx - 4 * mad.lsdx
  threshold.lsdy_low <- median.lsdy - 4 * mad.lsdy
  threshold.lsdx_high <- median.lsdx + 4 * mad.lsdx
  threshold.lsdy_high <- median.lsdy + 4 * mad.lsdy

  # Determine if event is a blink, fixation, or too dispersed
  event <- ifelse(
    (!is.na(lsdx) & lsdx < threshold.lsdx_low) &
      (!is.na(lsdy) & lsdy < threshold.lsdy_low),
    "blink", "fixation"
  )
  event <- ifelse(
    (!is.na(lsdx) & lsdx > threshold.lsdx_high) &
      (!is.na(lsdy) & lsdy > threshold.lsdy_high),
    "too dispersed", event
  )

  # Duration threshold for artifacts
  dur_inv <- 1 / events$dur
  median.dur <- stats::median(dur_inv, na.rm = TRUE)
  mad.dur <- stats::mad(dur_inv, na.rm = TRUE)
  threshold.dur <- median.dur + 5 * mad.dur

  # Label events as "too short" if necessary
  event <- ifelse(event != "blink" & dur_inv > threshold.dur, "too short", event)

  # Add event labels to the dataframe
  events$event <- factor(event, levels = c("fixation", "blink", "too dispersed", "too short"))

  return(events)
}

detect.saccades <- function(samples, lambda, smooth.saccades) {
  
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
  if (smooth.saccades) {
    sacc <- stats::filter(sacc, rep(1/3, 3))
    sacc <- as.logical(round(sacc))
  }
  samples$saccade <- ifelse(is.na(sacc), FALSE, sacc)
  samples$vx <- vx
  samples$vy <- vy

  samples
  
}

# SACCADE DETECTION
detect.and.process.saccades <- function(samples, lambda = 6) {
  
  # Calculate horizontal and vertical velocities:
  vx <- stats::filter(samples$x, -1:1/2)
  vy <- stats::filter(samples$y, -1:1/2)
  
  # Compute median-based standard deviation
  msdx <- sqrt(stats::median(vx^2, na.rm=TRUE) - stats::median(vx, na.rm=TRUE)^2)
  msdy <- sqrt(stats::median(vy^2, na.rm=TRUE) - stats::median(vy, na.rm=TRUE)^2)
  
  # Define thresholds
  radiusx <- msdx * lambda
  radiusy <- msdy * lambda
  
  # Identify saccades
  sacc <- ((vx/radiusx)^2 + (vy/radiusy)^2) > 1
  samples$saccade <- ifelse(is.na(sacc), FALSE, sacc)
  samples$vx <- vx
  samples$vy <- vy
  
  # Detect saccade changes
  saccade_events <- diff(c(FALSE, samples$saccade))
  saccade_starts <- which(saccade_events == 1)
  rle_sacc <- rle(samples$saccade)
  saccade_ends <- cumsum(rle_sacc$lengths)[rle_sacc$values == TRUE]
  
  # Initialize columns for start and end times/coordinates
  samples$start_time <- NA
  samples$start_x <- NA
  samples$start_y <- NA
  samples$end_time <- NA
  samples$end_x <- NA
  samples$end_y <- NA
  samples$start_trial <- NA
  samples$paragraph <- NA
  samples$participant <- NA

  # Assign start times/coordinates
  samples$start_time[saccade_starts] <- samples$time[saccade_starts]
  samples$start_x[saccade_starts] <- samples$x[saccade_starts]
  samples$start_y[saccade_starts] <- samples$y[saccade_starts]
  samples$start_trial[saccade_starts] <- samples$trial[saccade_starts]
  
  # Assign end times/coordinates
  samples$end_time[saccade_ends] <- samples$time[saccade_ends]
  samples$end_x[saccade_ends] <- samples$x[saccade_ends]
  samples$end_y[saccade_ends] <- samples$y[saccade_ends]
  
  # Calculate saccade metrics
  saccade_list <- list()
  for (i in 1:length(saccade_starts)) {
    start_idx <- saccade_starts[i]
    end_idx <- saccade_ends[i]
    
    # Ensure indices are valid
    if (!is.na(start_idx) && !is.na(end_idx) &&
        start_idx > 0 && end_idx > 0 &&
        start_idx <= nrow(samples) && end_idx <= nrow(samples)) {
      
      start_x <- samples$start_x[start_idx]
      start_y <- samples$start_y[start_idx]
      end_x <- samples$end_x[end_idx]
      end_y <- samples$end_y[end_idx]
      
      amplitude <- sqrt((end_x - start_x)^2 + (end_y - start_y)^2)
      angle <- atan2(end_y - start_y, end_x - start_x) * (180 / pi)
      
      saccade_list[[i]] <- data.frame(
        trial = samples$start_trial[start_idx],
        paragraph = samples$Paragraph[start_idx],  
        paragraph = samples$Participant[start_idx],
        start_time = samples$start_time[start_idx],
        end_time = samples$end_time[end_idx],
        duration = samples$end_time[end_idx] - samples$start_time[start_idx],
        start_x = start_x,
        start_y = start_y,
        end_x = end_x,
        end_y = end_y,
        amplitude = amplitude,
        angle = angle
      )
    }
  }
  
  # Combine the saccade metrics into a data frame
  saccades_df <- do.call(rbind, saccade_list)
  
  return(saccades_df)
}

''')

start = time.time()
detect_and_process_events = robjects.globalenv['detect.and.process.events']
detect_and_process_saccades = robjects.globalenv['detect.and.process.saccades']
end = time.time()
print(f"Done loading R objects. Took {end - start} seconds.")


def detect_events(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    start_time = time.time()
    # convert Pandas to R DataFrame
    r_df = robjects.pandas2ri.py2rpy(df)

    events_list = detect_and_process_events(r_df)
    fixations_r_df = events_list.rx2('fixations')
    blinks_r_df = events_list.rx2('blinks')
    saccades_r_df = detect_and_process_saccades(r_df)

    # convert R DataFrame back to Pandas
    fixations_df = pandas2ri.rpy2py(fixations_r_df)
    blinks_df = pandas2ri.rpy2py(blinks_r_df)
    saccades_df = pandas2ri.rpy2py(saccades_r_df)
    end_time = time.time()
    print(f"Done processing window for event detection. Took {end_time - start_time} seconds.")

    return fixations_df, blinks_df, saccades_df
