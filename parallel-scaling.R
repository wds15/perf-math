library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)

theme_set(theme_bw())

runcsv <- dir(pattern="parallel_threads_[0-9]+.csv")

names(runcsv) <- gsub("^(parallel_threads_)([0-9]+)(.csv)$", "\\2", runcsv)

rundata <- lapply(runcsv, read_csv) %>% bind_rows(.id="threads") %>%
    group_by(threads, name) %>%
    summarise(m=mean(real_time) * 1E-9 * 1E3) %>%
    separate(name, c("dummy", "algo", "case")) %>% select(-dummy)


rundata  <- rundata %>%
    group_by(algo, case) %>%
    mutate(threads=as.numeric(threads), speedup=m[threads == 1] / m) %>%
    ungroup() %>%
    as.data.frame()

head(rundata)
tail(rundata)

ggplot(rundata, aes(threads, m, colour=case, shape=algo)) + geom_point() + geom_line() +
    scale_x_log10(breaks=c(1,2,4,6)) +
    scale_y_log10() +
    ylab(NULL) +
    ggtitle("Execution time in milliseconds", "10^4 grouped poisson log-lik terms; dense = 10 groups; sparse = 100 groups")

ggsave("execution_time.png", width=1.62*4, height=4)

ggplot(rundata, aes(threads, speedup, colour=case, shape=algo)) + geom_point(alpha=0.3) + geom_line(alpha=0.3) +
    geom_text(aes(label=paste0(round(speedup, 1), "x"))) +
    scale_x_log10(breaks=c(1,2,4,6)) +
    scale_y_log10() +
    ylab(NULL) +
    ggtitle("Speedup vs 1 core", "10^4 grouped poisson log-lik terms; dense = 10 groups; sparse = 100 groups")

ggsave("execution_speedup.png", width=1.62*4, height=4)
