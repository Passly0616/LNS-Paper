param(
    [int]$Instances = 20,
    [int]$Repeats = 3,
    [int]$BudgetMs = 300,
    [uint32]$Seed = 20260524,
    [string]$K = "16,24,32,40,48,64",
    [string]$Out = "results"
)

$ErrorActionPreference = "Stop"

Write-Host "Compiling improved_experiments.cpp ..."
g++ -std=c++17 -O2 improved_experiments.cpp -o improved_experiments.exe

Write-Host "Running improved experiments ..."
.\improved_experiments.exe --instances $Instances --repeats $Repeats --budget-ms $BudgetMs --seed $Seed --k $K --out $Out

Write-Host "Done. Results are in $Out"
