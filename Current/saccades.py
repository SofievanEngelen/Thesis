import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd


pandas2ri.activate()


# Define the R function in R
robjects.r('''
    detect.fixations <- function(samples, lambda=6, smooth.coordinates=FALSE, smooth.saccades=TRUE) {
        if (! all(c("x", "y", "trial", "time") %in% colnames(samples)))
            stop("Input data frame needs columns 'x', 'y', 'trial', and 'time'.")
        
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
        event <- ifelse((!is.na(lsdx) & lsdx < threshold.lsdx) & (!is.na(lsdy) & lsdy < threshold.lsdy), "blink", "fixation")

        # Dispersion too high -> artifact:
        threshold.lsdx <- median.lsdx + 4 * mad.lsdx
        threshold.lsdy <- median.lsdy + 4 * mad.lsdy
        event <- ifelse((!is.na(lsdx) & lsdx > threshold.lsdx) & (!is.na(lsdy) & lsdy > threshold.lsdy), "too dispersed", event)

        # Artifact detection based on duration:
        dur <- 1/fixations$dur
        median.dur <- stats::median(dur, na.rm=TRUE)
        mad.dur <- stats::mad(dur, na.rm=TRUE)

        # Duration too short -> artifact:
        threshold.dur <- median.dur + mad.dur * 5
        event <- ifelse(event!="blink" & dur > threshold.dur, "too short", event)

        return(as.character(event))
        # factor(event, levels=c("fixation", "blink", "too dispersed", "too short"))
    }

    # This function takes a data frame of the samples and aggregates the
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

        fixations <- with(samples, data.frame(
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

    # Implementation of the Engbert & Kliegl algorithm for the
    # detection of saccades.  This function takes a data frame of the
    # samples and adds three columns:

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
''')

# Access the R function as a Python object
detect_fixations_func = robjects.globalenv['detect.fixations']


# Define the Python function that calls the R function
def detect_fixations(df: pd.DataFrame) -> pd.DataFrame:
    r_df = robjects.pandas2ri.py2rpy(df)  # Convert Pandas to R DataFrame
    result_r_df = detect_fixations_func(r_df)  # Call the R function

    # Convert the R DataFrame back to a Pandas DataFrame
    result_df = pandas2ri.rpy2py(result_r_df)
    return result_df  # Extract the result (as the result is wrapped in an R object)
