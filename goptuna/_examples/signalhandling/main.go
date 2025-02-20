package main

import (
	"context"
	"log"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"sync"
	"syscall"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"

	"github.com/mumax/3/goptuna"
	"github.com/mumax/3/goptuna/rdb.v2"
)

func objective(trial goptuna.Trial) (float64, error) {
	ctx := trial.GetContext()

	x1, _ := trial.SuggestFloat("x1", -10, 10)
	x2, _ := trial.SuggestFloat("x2", -10, 10)

	cmd := exec.CommandContext(ctx, "sleep", "1")
	err := cmd.Run()
	if err != nil {
		return -1, err
	}
	return math.Pow(x1-2, 2) + math.Pow(x2+5, 2), nil
}

func main() {
	db, err := gorm.Open(sqlite.Open("db.sqlite3"), &gorm.Config{})
	if err != nil {
		log.Fatal("failed to open database:", err)
	}
	if sqlDB, err := db.DB(); err != nil {
		log.Fatal("failed to get sql.DB:", err)
	} else {
		sqlDB.SetMaxOpenConns(1)
	}
	err = rdb.RunAutoMigrate(db)
	if err != nil {
		log.Fatal("failed to run auto migrate:", err)
	}

	// create a study
	study, err := goptuna.CreateStudy(
		"goptuna-example",
		goptuna.StudyOptionStorage(rdb.NewStorage(db)),
		goptuna.StudyOptionDirection(goptuna.StudyDirectionMinimize),
	)
	if err != nil {
		log.Fatal("failed to create a study:", err)
	}

	// create a context with cancel function
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	study.WithContext(ctx)

	// set signal handler
	sigch := make(chan os.Signal, 1)
	defer close(sigch)
	signal.Notify(sigch, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sig, ok := <-sigch
		if !ok {
			return
		}
		cancel()
		log.Println("Catch a kill signal:", sig.String())
	}()

	// run optimize with multiple goroutine workers
	concurrency := runtime.NumCPU() - 1
	if concurrency == 0 {
		concurrency = 1
	}
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err = study.Optimize(objective, 100/concurrency)
			if err != nil {
				log.Fatal("Optimize error:", err)
			}
		}()
	}
	wg.Wait()

	// print best hyper-parameters and the result
	v, _ := study.GetBestValue()
	params, _ := study.GetBestParams()
	log.Printf("Best evaluation=%f (x1=%f, x2=%f)",
		v, params["x1"].(float64), params["x2"].(float64))
}
