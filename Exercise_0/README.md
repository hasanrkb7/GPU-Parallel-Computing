\# Exercise 0 – Environment Setup (Proof Package)



\## Summary

This folder contains the required proof that the GitLab repository and GPU environment are correctly set up.



\## Repository setup

\- GitLab repository created and used for course submissions

\- Members added: kataka, tro

\- SSH key configured and repository accessed via SSH



\## GPU execution proof (CSC Puhti)

This exercise is considered complete only after running a GPU program on a university GPU server.



\### Files included

\- `job.sh` : SLURM batch script used on Puhti GPU node

\- `run\_output.txt` : output log produced by the job on Puhti (includes GPU info and successful run)



\## How to reproduce on Puhti

1\. Upload/copy your compiled binary to Puhti (example: `hello\_gpu`)

2\. Submit the job:

&nbsp;  ```bash

&nbsp;  sbatch job.sh



