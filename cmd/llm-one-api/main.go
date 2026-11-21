package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"llm-one-api/internal/relay"
)

func main() {
	path := flag.String("config", "config.yaml", "path to the YAML config file")
	flag.Parse()

	cfg, err := relay.LoadConfig(*path)
	if err != nil {
		log.Fatal(err)
	}
	srv := &http.Server{
		Addr:              cfg.Listen,
		Handler:           relay.New(cfg).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("listening on %s", cfg.Listen)
	log.Fatal(srv.ListenAndServe())
}
