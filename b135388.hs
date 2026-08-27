main = interact $ unlines . map f . lines
  where f line@('#':_) = line
        f line = unwords [show n, show $ a * product [1 .. n - 1] ^ (2 * n + 1)]
          where [n, a] = map read (words line) :: [Integer]
