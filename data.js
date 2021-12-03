//const { getItems, getInfo, getChoices, getQuestions } = require('@alheimsins/b5-johnson-120-ipip-neo-pi-r')
const { getItems, getInfo, getChoices, getQuestions } = require('@alheimsins/b5-costa-mccrae-300-ipip-neo-pi-r')
console.log(getQuestions()) // returns test info
const fs = require('fs')
fs.writeFile('./data/raw_questions.json', JSON.stringify(getQuestions()), err => {
  if (err) {
    console.error(err)
    return
  }
  //file written successfully
})
