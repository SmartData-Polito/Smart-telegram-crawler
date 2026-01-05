#!/bin/bash
# Run all TGDataset gaming experiments

echo "=============================================="
echo " TGDATASET GAMING EXPERIMENTS"
echo "=============================================="

# Step 0: Create seeds (separate, doesn't count in time)
echo ""
echo ">>> STEP 0: Creating seeds..."
python create_seeds.py

echo ""
echo ">>> Seeds created."

# Run experiments
for t in 20 40 60 80; do
    echo ""
    echo "=============================================="
    echo " THRESHOLD ${t}%"
    echo "=============================================="
    
    for type in pure mixed; do
        exp_name="threshold_${t}_${type}"
        
        echo ""
        echo "----------------------------------------------"
        echo " EXPERIMENT: ${exp_name}"
        echo "----------------------------------------------"
        
        # Start timer
        start_time=$(date +%s)
        
        # Run pipeline
        python master_orchestrator.py \
            --experiment-name ${exp_name} \
            --threshold 0.${t}
        
        # End timer
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        
        echo ""
        echo ">>> ${exp_name} completed in ${elapsed} seconds"
        
        # Analyze
        python analyze_all_levels.py \
            --experiment-name ${exp_name} \
            --threshold 0.${t}
    done
    
    # Pausa dopo ogni coppia pure/mixed
    echo ""
    echo "=============================================="
    echo " THRESHOLD ${t}% COMPLETED (pure + mixed)"
    echo "=============================================="
    echo ""
    
    if [ "$t" != "80" ]; then
        echo ">>> Press Enter to continue with next threshold..."
        read
    fi
done

# Final comparison
echo ""
echo "=============================================="
echo " FINAL COMPARISON"
echo "=============================================="
python analyze_all_levels.py --all

echo ""
echo ">>> ALL EXPERIMENTS COMPLETED"
```

---

**Comportamento:**
```
THRESHOLD 20%
  → threshold_20_pure (esegue)
  → threshold_20_mixed (esegue)
  → "Press Enter to continue..." (PAUSA)

THRESHOLD 40%
  → threshold_40_pure (esegue)
  → threshold_40_mixed (esegue)
  → "Press Enter to continue..." (PAUSA)

THRESHOLD 60%
  → threshold_60_pure (esegue)
  → threshold_60_mixed (esegue)
  → "Press Enter to continue..." (PAUSA)

THRESHOLD 80%
  → threshold_80_pure (esegue)
  → threshold_80_mixed (esegue)
  → (nessuna pausa, è l'ultimo)

FINAL COMPARISON