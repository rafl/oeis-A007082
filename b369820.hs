main = interact $ unlines . map f . lines
  where f line@('#':_) = line
        f line = unwords [show n, show $ a `div` 2]
          where [n, a] = map (read @Integer) (words line)
