# oneneuron
project_1

'''bash
git add. && git commit -m "docstring updated" && git push origin main
'''

## Add URL -
[Git Handbook](https://guides.github.com/introduction/git-handbook/)

<a href="https://www.w3schools.com">Visit W3Schools.com!</a>

## Add image -
![sample image](plots/and.png)

<img src="plots/and.png" alt=and.png" width="500" height="600">

```python
def main(data, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(data)

    logging.info(f"This is actual dataframe {df}")

    X, y = prepare_data(df)


    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plotFileName, model)


if __name__ == '__main__':  # << entry point
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1]
    }
    ETA = 0.3   # between 0 and 1
    EPOCHS = 10
    try:
        loggin.info(">>>> starting training >>>>")
        main(data=AND, eta=ETA, epochs=EPOCHS, filename='and.model', plotFileName='and.png', test=22)  # test included to test logging
        loggin.info("<<<< training done successfully <<<<")
    except Exception as e: 
        logging.exception(e)
        raise e 
```

