import pandas as pd
import numpy as np

"""
#' Functions for the detection of fixations in raw eye-tracking Data.
#'
#' Offers a function for detecting fixations in a stream of eye
#' positions recorded by an eye-tracker.  The detection is done using
#' an algorithm for saccade detection proposed by Ralf Engbert and
#' Reinhold Kliegl (see reference below).  Anything that happens
#' between two saccades is considered to be a fixation.  This software
#' is therefore not suited for Data sets with smooth-pursuit eye
#' movements.
#'
#' @name saccades
#' @docType package
#' @title Detection of Fixations in Raw Eye-Tracking Data
#' @author Titus von der Malsburg \email{malsburg@@posteo.de}
#' @references
#' Ralf Engbert, Reinhold Kliegl: Microsaccades uncover the
#' orientation of covert attention, Vision Research, 2003.
#' @importFrom zoom zm
#' @keywords manip ts classif
#' @seealso \code{\link{detect.fixations}},
#' \code{\link{diagnostic.plot}}, \code{\link{calculate.summary}}

NULL

#' Samples of eye positions as recorded with an iViewX eye-tracker
#' recording at approx. 250 Hz.  The Data quality is low on purpose
#' and contains episodes of track-loss and blinks.
#'
#' @name samples
#' @title Samples of Eye Positions as Recorded with an Eye-Tracker
#' @docType Data
#' @usage samples
#' @format a Data frame containing one line per sample.  The samples
#' are sorted in chronological order.  Time is given in milliseconds,
#' x- and y-coordinates in screen pixels.
#' @source Recorded with an iViewX Eye-Tracker by SMI at approximately
#' 250 Hz.
#' @author Titus von der Malsburg \email{malsburg@@posteo.de}

NULL

#' Fixations detected in a stream of raw eye positions.  The
#' corresponding raw eye positions samples are found in the Data frame
#' \code{\link{samples}} also part of this package.
#'
#' @name fixations
#' @title Fixations Detected in a Stream of Raw Positions
#' @docType Data
#' @usage fixations
#' @format a Data frame containing one line per fixation.  The
#' fixations are sorted in chronological order.  Time is given in
#' milliseconds, x- and y-coordinates in screen pixels.
#' @source Recorded with an iViewX Eye-Tracker by SMI at approximately
#' 250 Hz.
#' @author Titus von der Malsburg \email{malsburg@@posteo.de}

NULL

#' Takes a Data frame containing raw eye-tracking samples and returns a
#' Data frame containing fixations.
#'
#' @title Detect Fixations in a Stream of Raw Eye-Tracking Samples
#' @param samples a Data frame containing the raw samples as recorded
#' by the eye-tracker.  This Data frame has four columns:
#' \describe{
#'  \item{time:}{the time at which the sample was recorded}
#'  \item{trial:}{the trial to which the sample belongs}
#'  \item{x:}{the x-coordinate of the sample}
#'  \item{y:}{the y-coordinate of the sample}
#' }
#' Samples have to be listed in chronological order.  The velocity
#' calculations assume that the sampling frequency is constant.
#' @param lambda a parameter for tuning the saccade
#' detection.  It specifies which multiple of the standard deviation
#' of the velocity distribution should be used as the detection
#' threshold.
#' @param smooth.coordinates logical. If true the x- and y-coordinates will be
#' smoothed using a moving average with window size 3 prior to saccade
#' detection.
#' @param smooth.saccades logical.  If true, consecutive saccades that
#' are separated only by a few samples will be joined.  This avoids
#' the situation where swing-backs at the end of longer saccades are
#' recognized as separate saccades.  Whether this works well, depends
#' to some degree on the sampling rate of the eye-tracker.  If the
#' sampling rate is very high, the gaps between the main saccade and
#' the swing-back might become too large and look like genuine
#' fixations.  Likewise, if the sampling frequency is very low,
#' genuine fixations may be regarded as spurious.  Both cases are
#' unlikely to occur with current eye-trackers.
#' @section Details: This function uses a velocity-based detection
#' algorithm for saccades proposed by Engbert and Kliegl.  Anything
#' between two saccades is considered to be a fixation.  Thus, the
#' algorithm is not suitable for Data sets containing episodes of
#' smooth pursuit eye movements.
#' @return a Data frame containing the detected fixations.  This Data
#' frame has the following columns:
#'  \item{trial}{the trial to which the fixation belongs}
#'  \item{start}{the time at which the fixation started}
#'  \item{end}{the time at which the fixation ended}
#'  \item{x}{the x-coordinate of the fixation}
#'  \item{y}{the y-coordinate of the fixation}
#'  \item{mad.x}{the standard deviation of the sample x-coordinates within the fixation}
#'  \item{mad.y}{the standard deviation of the sample y-coordinates within the fixation}
#'  \item{peak.vx}{the horizontal peak velocity that was reached within the fixation}
#'  \item{peak.vy}{the vertical peak velocity that was reached within the fixation}
#'  \item{dur}{the duration of the fixation}
#'  \item{event}{the type of event, which could be 'fixation',
#'   'blink', or artifacts which are labeled 'too dispersed' and 'too
#'   short'.  Classification is based on simple heuristics that
#'   identify outliers with respect to dispersion and duration.}
#' @author Titus von der Malsburg \email{malsburg@@posteo.de}
#' @references
#' Ralf Engbert, Reinhold Kliegl: Microsaccades uncover the
#' orientation of covert attention, Vision Research, 2003.
#' @keywords manip ts classif
#' @seealso \code{\link{diagnostic.plot}},
#' \code{\link{calculate.summary}}
#' @export
#' @examples
#' Data(samples)
#' head(samples)
#' fixations <- detect.fixations(samples)
#' head(fixations)
#' \dontrun{
#' first.trial <- samples$trial[1]
#' first.trial.samples <- subset(samples, trial==first.trial)
#' first.trial.fixations <- subset(fixations, trial==first.trial)
#' with(first.trial.samples, plot(x, y, pch=20, cex=0.2, col="red"))
#' with(first.trial.fixations, points(x, y, cex=1+sqrt(dur/10000)))
#' }
detect.fixations <- function(samples, lambda=6, smooth.coordinates=FALSE, smooth.saccades=TRUE) {

  if (! all(c("x", "y", "trial", "time") %in% colnames(samples)))
    stop("Input Data frame needs columns 'x', 'y', 'trial', and 'time'.")

  if (! all(with(samples, tapply(time, trial, function(x) all(diff(x) > 0)))))
    stop("Samples need to be in chronological order within trial.")

  # Discard unnecessary columns:
  samples <- samples[c("x", "y", "trial", "time")]

  if (smooth.coordinates) {
    # Keep and reuse original first and last coordinates as they can't
    # be smoothed:
    x <- samples$x[c(1,nrow(samples))]
    y <- samples$y[c(1,nrow(samples))]
    kernel <- rep(1/3, 3)
    samples$x <- stats::filter(samples$x, kernel)
    samples$y <- stats::filter(samples$y, kernel)
    # Plug in the original values:
    samples$x[c(1,nrow(samples))] <- x
    samples$y[c(1,nrow(samples))] <- y
  }

  samples <- detect.saccades(samples, lambda, smooth.saccades)

  if (all(!samples$saccade))
    stop("No saccades were detected.  Something went wrong.")

  fixations <- aggregate.fixations(samples)

  fixations$event <- label.blinks.artifacts(fixations)

  fixations

}
"""

# PER PARTICIPANT !!!


def detect_fixations(samples: pd.DataFrame, lambd: int = 6, smooth_coordinates: bool = False,
                     smooth_saccades: bool = True) -> pd.DataFrame:
    if not all([col in samples.columns for col in ["x", "y", "trial", "time"]]):
        raise ValueError("Input Data frame needs columns 'x', 'y', 'trial', and 'time'.")

    if not all([all(np.diff(samples.query(f"trial == {trial}")["time"]) > 0) for trial in samples["trial"].unique()]):
        raise ValueError("Samples need to be in chronological order within trial.")

    samples = samples[["x", "y", "trial", "time"]]

    if smooth_coordinates:
        x = samples["x"].iloc[[0, -1]]
        y = samples["y"].iloc[[0, -1]]
        kernel = np.array([1 / 3, 1 / 3, 1 / 3])
        samples["x"] = np.convolve(samples["x"], kernel, mode="same")
        samples["y"] = np.convolve(samples["y"], kernel, mode="same")
        samples["x"].iloc[[0, -1]] = x
        samples["y"].iloc[[0, -1]] = y

    samples = detect_saccades(samples, lambd, smooth_saccades)

    if all(~samples["saccade"]):
        raise ValueError("No saccades were detected. Something went wrong.")

    fixations = aggregate_fixations(samples)

    fixations["event"] = label_blinks_artifacts(fixations)

    return fixations


"""

# EXPERIMENTAL: This function tries to detect blinks and artifacts
# based on x- and y-dispersion and duration of fixations.
label.blinks.artifacts <- function(fixations) {

  # Blink and artifact detection based on dispersion:
  lsdx <- log10(fixations$mad.x)
  lsdy <- log10(fixations$mad.y)
  median.lsdx <- stats::median(lsdx, na.rm=TRUE)
  median.lsdy <- stats::median(lsdy, na.rm=TRUE)
  mad.lsdx <- stats::mad(lsdx, na.rm=TRUE)
  mad.lsdy <- stats::mad(lsdy, na.rm=TRUE)

  # Dispersion too low -> blink:
  threshold.lsdx <- median.lsdx - 4 * mad.lsdx
  threshold.lsdy <- median.lsdy - 4 * mad.lsdy
  event <- ifelse((!is.na(lsdx) & lsdx < threshold.lsdx) &
                  (!is.na(lsdy) & lsdy < threshold.lsdy),
                  "blink", "fixation")

  # Dispersion too high -> artifact:
  threshold.lsdx <- median.lsdx + 4 * mad.lsdx
  threshold.lsdy <- median.lsdy + 4 * mad.lsdy
  event <- ifelse((!is.na(lsdx) & lsdx > threshold.lsdx) &
                  (!is.na(lsdy) & lsdy > threshold.lsdy),
                  "too dispersed", event)

  # Artifact detection based on duration:
  dur <- 1/fixations$dur
  median.dur <- stats::median(dur, na.rm=TRUE)
  mad.dur <- stats::mad(dur, na.rm=TRUE)

  # Duration too short -> artifact:
  threshold.dur <- median.dur + mad.dur * 5
  event <- ifelse(event!="blink" & dur > threshold.dur, "too short", event)

  factor(event, levels=c("fixation", "blink", "too dispersed", "too short"))
}
"""


def label_blinks_artifacts(fixations: pd.DataFrame) -> np.array:
    lsdx = np.log10(fixations["mad.x"])
    lsdy = np.log10(fixations["mad.y"])
    median_lsdx = np.median(lsdx)
    median_lsdy = np.median(lsdy)
    mad_lsdx = np.median(np.abs(lsdx - np.median(lsdx)))
    mad_lsdy = np.median(np.abs(lsdy - np.median(lsdy)))

    threshold_lsdx = median_lsdx - 4 * mad_lsdx
    threshold_lsdy = median_lsdy - 4 * mad_lsdy
    event = np.where((~np.isnan(lsdx) & (lsdx < threshold_lsdx)) &
                     (~np.isnan(lsdy) & (lsdy < threshold_lsdy)), "blink", "fixation")

    threshold_lsdx = median_lsdx + 4 * mad_lsdx
    threshold_lsdy = median_lsdy + 4 * mad_lsdy
    event = np.where((~np.isnan(lsdx) & (lsdx > threshold_lsdx)) &
                     (~np.isnan(lsdy) & (lsdy > threshold_lsdy)), "too dispersed", event)

    dur = 1 / fixations["dur"]
    median_dur = np.median(dur)
    mad_dur = np.median(np.abs(dur - np.median(dur)))

    threshold_dur = median_dur + mad_dur * 5
    event = np.where(event != "blink" & (dur > threshold_dur), "too short", event)

    return event


"""

# This function takes a Data frame of the samples and aggregates the
# samples into fixations.  This requires that the samples have been
# annotated using the function detect.saccades.
aggregate.fixations <- function(samples) {

  # In saccade.events a 1 marks the start of a saccade and a -1 the
  # start of a fixation.

  saccade.events <- sign(c(0, diff(samples$saccade)))

  trial.numeric  <- as.integer(factor(samples$trial))
  trial.events   <- sign(c(0, diff(trial.numeric)))

  # New fixations start either when a saccade ends or when a trial
  # ends:
  samples$fixation.id <- cumsum(saccade.events==-1|trial.events==1)
  samples$t2 <- samples$time
  samples$t2 <- ifelse(trial.events==1, NA, samples$t2)
  samples$t2 <- samples$t2[2:(nrow(samples)+1)]
  # Set last t2 value in a trial to last time value to avoid -Inf dur
  # and end values when the last event has just one sample (see #13 on
  # Github).  May produce zero-duration events but zero simply is our
  # most conservative guess in this case.
  samples$t2 <- with(samples, ifelse(is.na(t2), time, t2))

  # Discard samples that occurred during saccades:
  samples <- samples[!samples$saccade,,drop=FALSE]

  fixations <- with(samples, Data.frame(
    trial   = tapply(trial, fixation.id, function(x) x[1]),
    start   = tapply(time,  fixation.id, min),
    end     = tapply(t2,    fixation.id, function(x) max(x, na.rm=TRUE)),
    x       = tapply(x,     fixation.id, stats::median),
    y       = tapply(y,     fixation.id, stats::median),
    mad.x   = tapply(x,     fixation.id, stats::mad),
    mad.y   = tapply(y,     fixation.id, stats::mad),
    peak.vx = tapply(vx,    fixation.id, function(x) x[which.max(abs(x))]),
    peak.vy = tapply(vy,    fixation.id, function(x) x[which.max(abs(x))]),
    stringsAsFactors=FALSE))

  fixations$dur <- fixations$end - fixations$start

  fixations

}
"""


def aggregate_fixations(samples: pd.DataFrame) -> pd.DataFrame:
    saccade_events = np.sign(np.concatenate(([0], np.diff(samples["saccade"]))))

    trial_numeric = samples["trial"].factorize()[0]
    trial_events = np.sign(np.concatenate(([0], np.diff(trial_numeric))))

    samples["fixation.id"] = np.cumsum((saccade_events == -1) | (trial_events == 1))
    samples["t2"] = samples["time"]
    samples["t2"] = np.where(trial_events == 1, np.nan, samples["t2"])
    samples["t2"] = samples["t2"].shift(-1)
    samples["t2"] = np.where(samples["t2"].isna(), samples["time"], samples["t2"])

    samples = samples[~samples["saccade"]]

    fixations = samples.groupby("fixation.id").agg(
        trial=("trial", "first"),
        start=("time", "min"),
        end=("t2", lambda x: x.max(skipna=True)),
        x=("x", "median"),
        y=("y", "median"),
        mad_x=("x", "mad"),
        mad_y=("y", "mad"),
        peak_vx=("vx", lambda x: x[np.argmax(np.abs(x))]),
        peak_vy=("vy", lambda x: x[np.argmax(np.abs(x))])
    ).reset_index()

    fixations["dur"] = fixations["end"] - fixations["start"]

    return fixations


"""

# Implementation of the Engbert & Kliegl algorithm for the
# detection of saccades.  This function takes a Data frame of the
# samples and adds three columns:
#
# - A column named "saccade" which contains booleans indicating
#   whether the sample occurred during a saccade or not.
# - Columns named vx and vy which indicate the horizontal and vertical
#   speed.
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

  msdx <- sqrt(stats::median(vx**2, na.rm=TRUE) - stats::median(vx, na.rm=TRUE)**2)
  msdy <- sqrt(stats::median(vy**2, na.rm=TRUE) - stats::median(vy, na.rm=TRUE)**2)

  radiusx <- msdx * lambda
  radiusy <- msdy * lambda

  sacc <- ((vx/radiusx)**2 + (vy/radiusy)**2) > 1
  if (smooth.saccades) {
    sacc <- stats::filter(sacc, rep(1/3, 3))
    sacc <- as.logical(round(sacc))
  }
  samples$saccade <- ifelse(is.na(sacc), FALSE, sacc)
  samples$vx <- vx
  samples$vy <- vy

  samples

}
"""


def detect_saccades(samples, lambd: int, smooth_saccades: bool) -> pd.DataFrame:
    vx = samples["x"].rolling(window=3, center=True).mean()
    vy = samples["y"].rolling(window=3, center=True).mean()

    vx.iloc[0] = vx.iloc[1]
    vy.iloc[0] = vy.iloc[1]
    vx.iloc[-1] = vx.iloc[-2]
    vy.iloc[-1] = vy.iloc[-2]

    msdx = np.sqrt(np.median(vx ** 2) - np.median(vx) ** 2)
    msdy = np.sqrt(np.median(vy ** 2) - np.median(vy) ** 2)

    radiusx = msdx * lambd
    radiusy = msdy * lambd

    sacc = ((vx / radiusx) ** 2 + (vy / radiusy) ** 2) > 1
    if smooth_saccades:
        sacc = sacc.rolling(window=3, center=True).mean().round().astype(bool)

    samples["saccade"] = sacc
    samples["vx"] = vx
    samples["vy"] = vy

    return samples


