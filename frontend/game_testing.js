window.languagePluginUrl = 'pyodide/pyodide';
languagePluginLoader.then(function () {
    pyodide.loadPackage(['kale', 'numpy', 'pydantic', 'names']).then( () => {
        console.log(pyodide.runPython("\n" +
            "from game.game import Game, FakeClassifierPatientProvider\n" +
            "from game.constants import Disease\n" +
            "from game.util import confidences_df, treatment_effects_df\n" +
            "from kale.sampling.fake_clf import SufficientlyConfidentFC"))
        console.log(pyodide.runPython('\n' +
            'fake_clf = SufficientlyConfidentFC(len(list(Disease)))\n' +
            'patient_provider = FakeClassifierPatientProvider(fake_clf)\n' +
            'game = Game(patient_provider)\n' +
            'game.start_new_round(3)\n' +
            'game.play_current_round({\n' +
            '    game.current_round[0]: Disease.lung_cancer,\n' +
            '    game.current_round[1]: Disease.flu,\n' +
            '    game.current_round[2]: Disease.lung_cancer\n' +
            '})\n' +
            'print("Holy shit I cannot believe it actually works!")'));
    });
});