rsync -davzru --include-from <(git ls-files) --exclude .git --exclude-from <(git ls-files -o --directory) . mnj:/mnt/vol21/i20_udoooom/src/darts/
