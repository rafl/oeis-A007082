main = interact $ unlines . map f . lines
  where f line@('#':_) = line
        f line = unwords [ show n, show $ (n+1) * (2*n + 1) * n^(2*n + 1) * a ]
          where [n, a] = map (read @Integer) (words line)
