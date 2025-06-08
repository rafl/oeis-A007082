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
import qualified Data.Vector as V
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

numRoots :: Natural -> Prime Natural -> Natural
numRoots m p = gcd m (unPrime p - 1)

numPrimitiveRoots :: Natural -> Prime Natural -> Natural
numPrimitiveRoots m p
  | (unPrime p - 1) `mod` m == 0 = totient m
  | otherwise = 0

mFor :: Natural -> Natural
mFor n = head $ filter odd [(n+1) `div` 2, (n+3) `div` 2]

createMultisetStems :: Int -> Int -> [[Int]]
createMultisetStems 1 r = [[a] | a <- [0..r]]
createMultisetStems n 0 = [replicate n 0]
createMultisetStems n r = [a : b | a <- [0..r], b <- (createMultisetStems (n-1) (r - a))]

createMultisets :: Int -> Int -> [[Int]]
createMultisets 1 r = [[r]]
createMultisets n 0 = [replicate n 0]
createMultisets n r = [a : b | a <- [0..r], b <- (createMultisets (n-1) (r - a))]

expandExponents :: [Int] -> V.Vector Int
expandExponents multiset = 
  V.fromListN (sum multiset) (concat . map (uncurry replicate) $ zip multiset [0..]) 

matrixDeterminant :: [[SomeMod]] -> SomeMod
matrixDeterminant ass = go 0 ass 1
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

generateMatrixFromPreComputed :: Int -> V.Vector Int -> V.Vector (V.Vector SomeMod) -> V.Vector (V.Vector SomeMod) -> [[SomeMod]]
generateMatrixFromPreComputed n exponents jkPairs jkSumInverses = 
  [ [ a j k | k <- [0..n-2] ] | j <- [0..n-2] ] 
    where a j k
            | j /= k = negate $ ((jkPairs V.! (exponents V.! j)) V.! (exponents V.! k)) * ((jkSumInverses V.! (exponents V.! j)) V.! (exponents V.! k))
            | otherwise = sum [ ((jkPairs V.! (exponents V.! j)) V.! (exponents V.! r)) * ((jkSumInverses V.! (exponents V.! j)) V.! (exponents V.! r))
                          | r <- [0..n-1]
                          , r /= j
                      ]

compDeterminant :: Int -> V.Vector Int -> V.Vector (V.Vector SomeMod) -> V.Vector (V.Vector SomeMod) -> SomeMod
compDeterminant n exponents jkPairs jkSumInverses = 
  matrixDeterminant $ generateMatrixFromPreComputed n exponents jkPairs jkSumInverses
  -- matrixDeterminant . dropLast $ generateMatrixFromPreComputed n jkPairs jkSumInverses
  --   where dropLast = init . map init

compCoef :: [Int] -> Int
compCoef multiset = 
    product [1..(sum multiset)] `div` product (map (\m -> product [1..m]) multiset)

f :: Int -> Natural -> V.Vector Int -> V.Vector SomeMod -> V.Vector SomeMod -> V.Vector (V.Vector SomeMod) -> V.Vector (V.Vector SomeMod) -> V.Vector (V.Vector SomeMod) -> SomeMod
f n m exponents mthRoots mthRootInverses jkPairs jkSums jkSumInverses = 
  -- traceShowWith ("exponents", exponents, ) $
  fstTrm * sndTrm
   where fstTrm = product [ (jkSums V.! (exponents V.! j)) V.! (exponents V.! k)
                          | (j:ks) <- tails [0..n-1]
                          , k <- ks ]
         sndTrm = compDeterminant n exponents jkPairs jkSumInverses

compCongruence :: Prime Natural -> Int -> Natural -> V.Vector SomeMod -> V.Vector SomeMod -> V.Vector (V.Vector SomeMod) -> V.Vector (V.Vector SomeMod) ->  V.Vector (V.Vector SomeMod) -> SomeMod
compCongruence p n m mthRoots mthRootInverses jkPairs jkSums jkSumInverses = 
  -- m^(-n+1)
    (powSomeMod (fromIntegral m `modulo` unPrime p) (-n+1)) * 
  -- parallelized sum of permutations, in exponent parameter space
  sum (map (`modulo` unPrime p) . runPar $ parMap 
    (\multiset ->  ((fromIntegral $ compCoef multiset)) * (read . show $ f n m (V.snoc (expandExponents multiset) 0) mthRoots mthRootInverses jkPairs jkSums jkSumInverses)) (createMultisets (fromIntegral m) (n-1)))
 
compCongruenceFromStems :: Prime Natural -> Int -> Natural -> V.Vector SomeMod -> V.Vector SomeMod -> V.Vector (V.Vector SomeMod) -> V.Vector (V.Vector SomeMod) ->  V.Vector (V.Vector SomeMod) -> SomeMod
compCongruenceFromStems p n m mthRoots mthRootInverses jkPairs jkSums jkSumInverses = 
  -- m^(-n+1)
    (powSomeMod (fromIntegral m `modulo` unPrime p) (-n+1)) * 
  -- parallelized sum of permutations, in exponent parameter space
  sum (map (`modulo` unPrime p) . runPar $ parMap 
    (\multisetStem -> read . show $ sum (map (\multiset -> ((fromIntegral $ compCoef (multisetStem ++ multiset)) * (f n m (V.snoc (expandExponents (multisetStem ++ multiset)) 0) mthRoots mthRootInverses jkPairs jkSums jkSumInverses))) (createMultisets (fromIntegral m - (fromIntegral m `div` 2)) (n-1-(sum multisetStem))))) (createMultisetStems (fromIntegral m `div` 2) (n-1)))
   

main :: IO ()
main = do
  args <- map (read @Natural) <$> getArgs
  let n = args !! 0
  -- let n = 7
  print ("n = " ++ show n)
  let m = mFor n
  print ("m = " ++ show m)
  let p = findPrime m 30
  print ("p = " ++ show p)
  let mthRoot = primitiveMthRoot m p
  print ("mthRoot = " ++ show mthRoot)
  let mthRoots = V.generate (fromIntegral m) (powSomeMod mthRoot)
  print ("mthRoots[] = " ++ show mthRoots)
  let mthRootInverses = V.map (\a -> (powSomeMod a (-1))) mthRoots
  print ("mthRootInverses[] = " ++ show mthRootInverses)
  let jkPairs = V.generate (fromIntegral m) (\j -> V.generate (fromIntegral m) (\k -> (mthRoots V.! j) * (mthRootInverses V.! k)))
  print "jkPairs[][]"
  putStr (unlines . map (("\t" ++) . show) . V.toList $ jkPairs)
  let jkSums = V.generate (fromIntegral m) (\j -> V.generate (fromIntegral m) (\k -> (jkPairs V.! j) V.! k + (jkPairs V.! k) V.! j))
  print "jkSums[][]"
  putStr (unlines . map (("\t" ++) . show) . V.toList $ jkSums)
  let jkSumInverses = V.generate (fromIntegral m) (\j -> V.generate (fromIntegral m) (\k -> powSomeMod ((jkSums V.! j) V.! k) (-1)))
  print "jkSumInverses[][]"
  putStr (unlines . map (("\t" ++) . show) . V.toList $ jkSumInverses)
  print (compCongruenceFromStems p (fromIntegral n) m mthRoots mthRootInverses jkPairs jkSums jkSumInverses)
  
-- print (e_n''' (fromIntegral n) p m w wm)