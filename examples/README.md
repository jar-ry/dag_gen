# R Version:

https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-18-04-quickstart
R 4.2.1

# Packages

## CAM

'install.packages("devtools")'
'library(devtools)'
'install_github("cran/CAM")'

## pcalg

'install.packages("BiocManager")'
'BiocManager::install(c("graph", "RBGL", "ggm", "fastICA"))'
'install.packages("pcalg")'
