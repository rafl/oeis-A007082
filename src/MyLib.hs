module MyLib (main) where

import Math.NumberTheory.Primes
import Math.NumberTheory.Moduli
import Math.NumberTheory.ArithmeticFunctions
import Data.Maybe
import Numeric.Natural
import Data.Mod
import Debug.Trace
import Control.Monad
import Control.Monad.Par
import Data.List (tails, sort, findIndex)
import qualified Data.Map as M
import System.Environment

findPrime :: Natural -> Int -> Prime Natural
findPrime m np = go 1
  where ps = take np primes
        base = product (map unPrime ps)
        go k = maybe (go (k+1)) id (isPrime $ k*base*m + 1)

{-# SCC primitiveMthRoot #-}
primitiveMthRoot :: Natural -> Prime Natural -> SomeMod
primitiveMthRoot m p = head
  [ g
  | a <- map (`modulo` unPrime p) [2..]
  , let g = powSomeMod a ((unPrime p - 1) `div` m)
  , g /= 1
  , all ((/= 1) . powSomeMod g . (m `div`)) mfacts
  ]
  where mfacts = map (unPrime . fst) (factorise m)

numRoots m p = gcd m (unPrime p - 1)
numPrimitiveRoots m p
  | (unPrime p - 1) `mod` m == 0 = totient m
  | otherwise = 0

f n m w xs = fstTrm * sndTrm
  where fstTrm = product [ (powSomeMod xj (-1)) * xk + xj * powSomeMod xk (-1)
                         | (j:ks) <- tails [0..n-1]
                         , k <- ks
                         , let xj = xs !! j
                         , let xk = xs !! k ]
        sndTrm = compDropDet n m w xs
		
f' n m w xs = read . show $ fstTrm * sndTrm
  where fstTrm = product [ (powSomeMod xj (-1)) * xk + xj * powSomeMod xk (-1)
                         | (j:ks) <- tails [0..n-1]
                         , k <- ks
                         , let xj = xs !! j
                         , let xk = xs !! k ]
        sndTrm = compDropDet n m w xs

compDropDet n m w xs = det' . dropLast $ compMat n m w xs
  where dropLast = init . map init

det [] = 1
det [[a]] = a
det ass@(as:_) =
  sum [ sign * (as !! j) * det (minor j ass)
      | j <- [0..length ass-1]
      , let sign | even j = 1
                 | otherwise = -1 ]

det' ass = go 0 ass 1
  where
    n = length ass
    go i mat acc
      | i == n    = acc
      | otherwise =
          case findIndex (\row -> row !! i /= 0) (drop i mat) of
            Nothing -> 0
            Just relJ ->
              let j = i + relJ
                  (mat', acc') = if j /= i
                                 then (swapRows i j mat, (-acc))
                                 else (mat, acc)
                  pivot = (mat' !! i) !! i
                  eliminateRow k m'
                    | k >= n    = m'
                    | otherwise =
                        let rowK = m' !! k
                            rowI = m' !! i
                            factor = (rowK !! i) / pivot
                            newRowK = zipWith (\a b -> a - factor * b) rowK rowI
                            m'' = replaceRow k newRowK m'
                        in eliminateRow (k + 1) m''

                  mat'' = eliminateRow (i + 1) mat'
              in go (i + 1) mat'' (acc' * pivot)

    swapRows r1 r2 xs =
      [ choose k row | (k,row) <- zip [0..] xs ]
      where
        choose k row
          | k == r1   = xs !! r2
          | k == r2   = xs !! r1
          | otherwise = row

    replaceRow i newRow xs =
      [ if k == i then newRow else row
      | (k,row) <- zip [0..] xs ]


minor j (_:as) = map (removeAt j) as
  where removeAt n as = take n as ++ drop (n+1) as


compMat n m w xs = [ [ a j k xj xk
                     | k <- [0..n-1]
                     , let xk = xs !! k ]
                   | j <- [0..n-1]
                   , let xj = xs !! j ]
  where a j k xj xk
          | j /= k = negate $ xj * (powSomeMod xk (-1)) * (powSomeMod ((powSomeMod xj (-1)) * xk + xj * powSomeMod xk (-1)) (-1))
          | otherwise = sum [ xj * (powSomeMod xr (-1)) * (powSomeMod ((powSomeMod xj (-1)) * xr + xj * powSomeMod xr (-1)) (-1))
                            | r <- [0..n-1]
                            , r /= j
                            , let xr = xs !! r ]

e_n n p m w =
  (powSomeMod (fromIntegral m `modulo` unPrime p) (-n+1)) * sum (map (f n m w) wss)
  where wss = traceShowId $ (++ [w^m]) . map (powSomeMod w) <$> replicateM (n-1) [0..m-1]

e_n' n p m w =
  (powSomeMod (fromIntegral m `modulo` unPrime p) (-n+1)) * sum (map ((cache M.!) . sort) wss)
  where wss = (++ [w^m]) . map (powSomeMod w) <$> replicateM (n-1) [0..m-1]
        cache = foldl' go M.empty wss
        go m ws | Just r <- M.lookup (sort ws) m = m
                | otherwise = M.insert (sort ws) (f n m w ws) m




-- createMss :: Integer -> [[Integer]]
-- createMss targetSum = [ rs | sum [rs] = targetSum, all rs >= 0 ]

createMss :: Int -> Int -> [[Int]]
createMss 1 r = [[r]]
createMss n 0 = [replicate n 0]
createMss n r = [a : b | a <- [0..r], b <- (createMss (n-1) (r - a)) ]

createWs :: SomeMod -> [Int] -> [SomeMod]
createWs w ms = 
	-- traceShowWith ("ws", ms, ) $
	map (powSomeMod w) . concat . map (uncurry replicate) $ zip ms [0..] 

compCoef :: [Int] -> Int
compCoef ms = 
	-- traceShowWith ("coef", ms, ) $
	product [1..(sum ms)] `div` product (map (\m -> product [1..m]) ms)


e_n'' n p m w =
  (powSomeMod (fromIntegral m `modulo` unPrime p) (-n+1)) * sum (map (`modulo` unPrime p) . runPar $ parMap (\ms -> (fromIntegral $ compCoef ms) * f' n m w ((createWs w ms) ++ [w^m])) (createMss (fromIntegral m) (n-1)))


	--    * sum (map
	--   	(map (compCoef(ms) * f n m w) wss where wss = (++ [w^m]) . map createWs w ms
	-- where ms = createMss (n-1) (n-1))
	--


main'' :: IO ()
main'' = do
  let m = 39
  let p = findPrime m 300
  print p
  let r = primitiveMthRoot m p
  print r
  print (r^2)
  print (r^m)
  print (r^(2*m))

mFor n = head $ filter odd [(n+1) `div` 2, (n+3) `div` 2]

main' :: IO ()
main' = do
  let (Just p) = isPrime 271
  let m = 3
  let w = primitiveMthRoot m p
  print (e_n'' 3 p m w)
  -- print (e_n 7 p m w)

main :: IO ()
main = do
  args <- map (read @Natural) <$> getArgs
  let n = args !! 0
  -- let n = 9
  print n
  let m = mFor n
  print m
  let p = findPrime m 30
  print p
  let w = primitiveMthRoot m p
  print w
  print (e_n'' (fromIntegral n) p m w)
 
test :: Par [Int]
test = parMap (^2) [1..10]


-- p = 271
-- n = 3 (or 5)
-- m = 3
-- 3rd-root mod 271 = 28
-- e_n 3 = 2
-- e_n 5 = 264
