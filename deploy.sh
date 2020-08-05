rsync -davzru --include-from <(git ls-files) --exclude .git --exclude-from <(git ls-files -o --directory) . udon:/home/udon/programing/aym/study/darts_homeworks/
